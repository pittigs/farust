//! GPU (`wgpu`) scene renderer supporting both interactive windows (`winit`) and
//! headless offscreen PNG snapshots.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::camera::OrbitCamera;
use crate::scene::{GpuVertex, LoadedScene};

pub(crate) const MSAA_SAMPLES: u32 = 4;
pub(crate) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    eye_pos: [f32; 4],
    sun_dir: [f32; 4],
}

const SHADER_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    sun_dir: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var t_albedo: texture_2d<f32>;
@group(0) @binding(2) var s_albedo: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.normal = in.normal;
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    let tex = textureSample(t_albedo, s_albedo, in.uv);
    let base_rgb = tex.rgb * in.color.rgb;
    let spec_strength = tex.a * in.color.a;

    var n = normalize(in.normal);
    if (!front_facing) {
        n = -n;
    }

    let l = normalize(u.sun_dir.xyz);
    let v = normalize(u.eye_pos.xyz - in.world_pos);
    let h = normalize(l + v);

    let hemi_t = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(vec3<f32>(0.15, 0.14, 0.13), vec3<f32>(0.34, 0.40, 0.50), hemi_t);

    let ndotl = max(dot(n, l), 0.0);
    let sun = vec3<f32>(1.15, 1.08, 0.96) * ndotl;

    let fill_dir = normalize(vec3<f32>(-l.x, 0.25, -l.z));
    let fill = vec3<f32>(0.16, 0.20, 0.28) * max(dot(n, fill_dir), 0.0);

    let spec = pow(max(dot(n, h), 0.0), 36.0) * spec_strength * (0.20 + 0.80 * ndotl);
    let lit = base_rgb * (ambient + sun + fill) + vec3<f32>(spec);
    return vec4<f32>(lit, 1.0);
}
"#;

pub(crate) struct GpuSceneResources {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    /// Per-frame geometry (units, overlays) drawn with the same pipeline.
    dynamic: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
}

impl GpuSceneResources {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &LoadedScene,
        target_format: wgpu::TextureFormat,
        sample_count: u32,
        camera: &OrbitCamera,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("farust_scene_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_WGSL.into()),
        });

        let tex_size = wgpu::Extent3d {
            width: scene.albedo.width.max(1),
            height: scene.albedo.height.max(1),
            depth_or_array_layers: 1,
        };
        let albedo_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("albedo_tex"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &albedo_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &scene.albedo.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * tex_size.width),
                rows_per_image: Some(tex_size.height),
            },
            tex_size,
        );
        let albedo_view = albedo_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("albedo_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniforms = make_uniforms(camera);
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform_buf"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buf"),
            contents: bytemuck::cast_slice(&scene.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buf"),
            contents: bytemuck::cast_slice(&scene.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            pipeline,
            bind_group,
            uniform_buf,
            vertex_buf,
            index_buf,
            index_count: scene.indices.len() as u32,
            dynamic: None,
        }
    }

    /// Replace the per-frame dynamic geometry (empty input clears it).
    pub(crate) fn set_dynamic(&mut self, device: &wgpu::Device, vertices: &[GpuVertex], indices: &[u32]) {
        if vertices.is_empty() || indices.is_empty() {
            self.dynamic = None;
            return;
        }
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dyn_vertex_buf"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dyn_index_buf"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.dynamic = Some((vb, ib, indices.len() as u32));
    }

    pub(crate) fn update_camera(&self, queue: &wgpu::Queue, camera: &OrbitCamera) {
        let uniforms = make_uniforms(camera);
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    pub(crate) fn encode_draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        resolve_target: Option<&wgpu::TextureView>,
        depth_view: &wgpu::TextureView,
        clear_color: [f64; 4],
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: clear_color[0],
                        g: clear_color[1],
                        b: clear_color[2],
                        a: clear_color[3],
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        rpass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        if self.index_count > 0 {
            rpass.draw_indexed(0..self.index_count, 0, 0..1);
        }
        if let Some((vb, ib, n)) = &self.dynamic {
            rpass.set_vertex_buffer(0, vb.slice(..));
            rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..*n, 0, 0..1);
        }
    }
}

fn make_uniforms(camera: &OrbitCamera) -> Uniforms {
    let eye = camera.eye_position();
    Uniforms {
        view_proj: camera.view_proj_matrix(),
        eye_pos: [eye[0], eye[1], eye[2], 1.0],
        sun_dir: [0.52, 0.76, 0.38, 0.0],
    }
}

pub(crate) fn create_attachment(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    sample_count: u32,
    usage: wgpu::TextureUsages,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// Renders `scene` offscreen on the GPU and writes an RGBA8 PNG to `out_path`.
pub fn render_scene_to_png(
    scene: &LoadedScene,
    width: u32,
    height: u32,
    out_path: &Path,
) -> Result<String, String> {
    if width == 0 || height == 0 {
        return Err("render dimensions must be non-zero".to_string());
    }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "no compatible Vulkan/GPU adapter found".to_string())?;

    let adapter_info = adapter.get_info();
    let (device, queue) = pollster::block_on(
        adapter.request_device(&wgpu::DeviceDescriptor::default(), None),
    )
    .map_err(|e| format!("failed to create GPU device: {e}"))?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut camera = scene.initial_camera.clone();
    camera.set_aspect(width, height);

    let resources = GpuSceneResources::new(&device, &queue, scene, format, MSAA_SAMPLES, &camera);

    let (_msaa_tex, msaa_view) = create_attachment(
        &device,
        width,
        height,
        format,
        MSAA_SAMPLES,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
        "offscreen_msaa",
    );
    let (resolve_tex, resolve_view) = create_attachment(
        &device,
        width,
        height,
        format,
        1,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        "offscreen_resolve",
    );
    let (_depth_tex, depth_view) = create_attachment(
        &device,
        width,
        height,
        DEPTH_FORMAT,
        MSAA_SAMPLES,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
        "offscreen_depth",
    );

    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let unpadded_bpr = width * 4;
    let padded_bpr = unpadded_bpr.div_ceil(align) * align;
    let readback_size = u64::from(padded_bpr) * u64::from(height);
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen_readback"),
        size: readback_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("offscreen_encoder"),
    });
    resources.encode_draw(
        &mut encoder,
        &msaa_view,
        Some(&resolve_view),
        &depth_view,
        scene.clear_color,
    );
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &resolve_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| format!("readback channel failed: {e}"))?
        .map_err(|e| format!("GPU buffer map failed: {e}"))?;

    let mapped = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((unpadded_bpr * height) as usize);
    for row in 0..height as usize {
        let start = row * padded_bpr as usize;
        pixels.extend_from_slice(&mapped[start..start + unpadded_bpr as usize]);
    }
    drop(mapped);
    readback_buf.unmap();

    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    let file = File::create(out_path)
        .map_err(|e| format!("failed to create {}: {e}", out_path.display()))?;
    let mut png_enc = png::Encoder::new(BufWriter::new(file), width, height);
    png_enc.set_color(png::ColorType::Rgba);
    png_enc.set_depth(png::BitDepth::Eight);
    let mut writer = png_enc
        .write_header()
        .map_err(|e| format!("png header error: {e}"))?;
    writer
        .write_image_data(&pixels)
        .map_err(|e| format!("png write error: {e}"))?;

    Ok(format!("{} ({:?})", adapter_info.name, adapter_info.backend))
}

/// Opens an interactive `winit` + `wgpu` viewer window with orbit, pan, and strategic zoom.
pub fn run_window(scene: LoadedScene, width: u32, height: u32) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| format!("failed to create event loop: {e}"))?;
    let mut app = ViewerApp {
        scene,
        init_size: (width.max(320), height.max(240)),
        state: None,
        init_err: None,
    };
    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("event loop error: {e}"))?;
    if let Some(err) = app.init_err {
        return Err(err);
    }
    Ok(())
}

struct WindowGpuState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    resources: GpuSceneResources,
    camera: OrbitCamera,
    left_drag: bool,
    right_drag: bool,
    last_cursor: Option<(f64, f64)>,
}

struct ViewerApp {
    scene: LoadedScene,
    init_size: (u32, u32),
    state: Option<WindowGpuState>,
    init_err: Option<String>,
}

impl ViewerApp {
    fn init_window(&mut self, event_loop: &ActiveEventLoop) -> Result<WindowGpuState, String> {
        let attrs = Window::default_attributes()
            .with_title(format!("FARust Viewer — {}", self.scene.title))
            .with_inner_size(PhysicalSize::new(self.init_size.0, self.init_size.1));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .map_err(|e| format!("failed to create window: {e}"))?,
        );

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("failed to create wgpu surface: {e}"))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| "no compatible GPU adapter for window surface".to_string())?;

        let (device, queue) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor::default(), None),
        )
        .map_err(|e| format!("failed to request GPU device: {e}"))?;

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let mut camera = self.scene.initial_camera.clone();
        camera.set_aspect(width, height);

        let resources =
            GpuSceneResources::new(&device, &queue, &self.scene, format, MSAA_SAMPLES, &camera);
        let (_msaa_tex, msaa_view) = create_attachment(
            &device,
            width,
            height,
            format,
            MSAA_SAMPLES,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            "win_msaa",
        );
        let (_depth_tex, depth_view) = create_attachment(
            &device,
            width,
            height,
            DEPTH_FORMAT,
            MSAA_SAMPLES,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            "win_depth",
        );

        Ok(WindowGpuState {
            window,
            surface,
            device,
            queue,
            config,
            msaa_view,
            depth_view,
            resources,
            camera,
            left_drag: false,
            right_drag: false,
            last_cursor: None,
        })
    }
}

impl ApplicationHandler for ViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            match self.init_window(event_loop) {
                Ok(st) => {
                    st.window.request_redraw();
                    self.state = Some(st);
                }
                Err(e) => {
                    self.init_err = Some(e);
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(st) = self.state.as_mut() else {
            return;
        };
        if st.window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    st.config.width = new_size.width;
                    st.config.height = new_size.height;
                    st.surface.configure(&st.device, &st.config);
                    let (_, msaa_view) = create_attachment(
                        &st.device,
                        new_size.width,
                        new_size.height,
                        st.config.format,
                        MSAA_SAMPLES,
                        wgpu::TextureUsages::RENDER_ATTACHMENT,
                        "win_msaa",
                    );
                    let (_, depth_view) = create_attachment(
                        &st.device,
                        new_size.width,
                        new_size.height,
                        DEPTH_FORMAT,
                        MSAA_SAMPLES,
                        wgpu::TextureUsages::RENDER_ATTACHMENT,
                        "win_depth",
                    );
                    st.msaa_view = msaa_view;
                    st.depth_view = depth_view;
                    st.camera.set_aspect(new_size.width, new_size.height);
                    st.resources.update_camera(&st.queue, &st.camera);
                    st.window.request_redraw();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => st.left_drag = pressed,
                    MouseButton::Right | MouseButton::Middle => st.right_drag = pressed,
                    _ => {}
                }
                if !st.left_drag && !st.right_drag {
                    st.last_cursor = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let cur = (position.x, position.y);
                if let Some((px, py)) = st.last_cursor {
                    let dx = (cur.0 - px) as f32;
                    let dy = (cur.1 - py) as f32;
                    if st.left_drag {
                        st.camera.orbit(-dx * 0.006, dy * 0.006);
                        st.resources.update_camera(&st.queue, &st.camera);
                        st.window.request_redraw();
                    } else if st.right_drag {
                        st.camera.pan(-dx * 0.0018, dy * 0.0018);
                        st.resources.update_camera(&st.queue, &st.camera);
                        st.window.request_redraw();
                    }
                }
                if st.left_drag || st.right_drag {
                    st.last_cursor = Some(cur);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 40.0) as f32,
                };
                if steps.abs() > 0.0 {
                    st.camera.zoom(steps);
                    st.resources.update_camera(&st.queue, &st.camera);
                    st.window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                let mut changed = true;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => {
                        event_loop.exit();
                        return;
                    }
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        let aspect = st.camera.aspect;
                        st.camera = self.scene.initial_camera.clone();
                        st.camera.aspect = aspect;
                    }
                    PhysicalKey::Code(KeyCode::KeyW | KeyCode::ArrowUp) => st.camera.pan(0.0, 0.06),
                    PhysicalKey::Code(KeyCode::KeyS | KeyCode::ArrowDown) => {
                        st.camera.pan(0.0, -0.06)
                    }
                    PhysicalKey::Code(KeyCode::KeyA | KeyCode::ArrowLeft) => {
                        st.camera.pan(-0.06, 0.0)
                    }
                    PhysicalKey::Code(KeyCode::KeyD | KeyCode::ArrowRight) => {
                        st.camera.pan(0.06, 0.0)
                    }
                    PhysicalKey::Code(KeyCode::KeyQ) => st.camera.orbit(-0.10, 0.0),
                    PhysicalKey::Code(KeyCode::KeyE) => st.camera.orbit(0.10, 0.0),
                    _ => changed = false,
                }
                if changed {
                    st.resources.update_camera(&st.queue, &st.camera);
                    st.window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                let frame = match st.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                        st.surface.configure(&st.device, &st.config);
                        return;
                    }
                    Err(_) => return,
                };
                let resolve_view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    st.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("win_encoder"),
                        });
                st.resources.encode_draw(
                    &mut encoder,
                    &st.msaa_view,
                    Some(&resolve_view),
                    &st.depth_view,
                    self.scene.clear_color,
                );
                st.queue.submit(Some(encoder.finish()));
                frame.present();
            }
            _ => {}
        }
    }
}
