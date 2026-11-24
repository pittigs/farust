use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use zip::ZipArchive;

#[derive(Debug, Clone)]
pub struct ScdEntry {
    pub name: String,
    pub size: u64,
    pub is_dir: bool,
}

pub fn list_scd(path: &Path) -> io::Result<Vec<ScdEntry>> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("not a valid .scd zip: {e}")))?;
    let mut entries = Vec::with_capacity(archive.len());
    for i in 0..archive.len() {
        let entry = archive.by_index(i).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("bad zip entry {i}: {e}"))
        })?;
        entries.push(ScdEntry {
            name: entry.name().to_string(),
            size: entry.size(),
            is_dir: entry.is_dir(),
        });
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

/// Read every non-directory entry whose name matches `filter`, in one open.
/// Returns `(inner_path, bytes)` sorted by path.
pub fn read_scd_matching(
    scd_path: &Path,
    filter: impl Fn(&str) -> bool,
) -> io::Result<Vec<(String, Vec<u8>)>> {
    let file = File::open(scd_path)?;
    let mut archive = ZipArchive::new(file)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("not a valid .scd zip: {e}")))?;
    let mut names: Vec<String> = Vec::new();
    for i in 0..archive.len() {
        if let Ok(entry) = archive.by_index(i) {
            let name = entry.name().to_string();
            if !name.ends_with('/') && filter(&name) {
                names.push(name);
            }
        }
    }
    names.sort();
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let mut entry = archive.by_name(&name).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("bad zip entry {name}: {e}"))
        })?;
        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buf)?;
        out.push((name, buf));
    }
    Ok(out)
}

pub fn read_scd_file(scd_path: &Path, inner_path: &str) -> io::Result<Vec<u8>> {
    let file = File::open(scd_path)?;
    let mut archive = ZipArchive::new(file)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("not a valid .scd zip: {e}")))?;
    let mut entry = archive.by_name(inner_path).map_err(|_| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("{inner_path} not found in {}", scd_path.display()),
        )
    })?;
    let mut buf = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Read an inner file case-insensitively (normalizing `/` and `\`), returning `(actual_name, bytes)`.
pub fn read_scd_file_ci(scd_path: &Path, inner_path: &str) -> io::Result<(String, Vec<u8>)> {
    let normalized = inner_path.trim_start_matches(['/', '\\']).replace('\\', "/");
    let file = File::open(scd_path)?;
    let mut archive = ZipArchive::new(file)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("not a valid .scd zip: {e}")))?;

    if let Ok(mut entry) = archive.by_name(&normalized)
        && !entry.is_dir()
    {
        let actual = entry.name().to_string();
        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buf)?;
        return Ok((actual, buf));
    }

    let mut matched_name: Option<String> = None;
    for i in 0..archive.len() {
        let entry = archive.by_index(i).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("bad zip entry {i}: {e}"))
        })?;
        if !entry.is_dir() && entry.name().eq_ignore_ascii_case(&normalized) {
            matched_name = Some(entry.name().to_string());
            break;
        }
    }

    let actual = matched_name.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("{inner_path} not found in {}", scd_path.display()),
        )
    })?;
    let mut entry = archive.by_name(&actual).map_err(|e| {
        io::Error::new(io::ErrorKind::InvalidData, format!("bad zip entry {actual}: {e}"))
    })?;
    let mut buf = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut buf)?;
    Ok((actual, buf))
}

pub fn default_game_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("FA_PATH") {
        return PathBuf::from(dir);
    }
    PathBuf::from("/home/bas_pit/.local/share/Steam/steamapps/common/Supreme Commander Forged Alliance")
}

pub fn gamedata_dir(game_dir: &Path) -> PathBuf {
    game_dir.join("gamedata")
}

pub fn maps_dir(game_dir: &Path) -> PathBuf {
    game_dir.join("maps")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_test_zip(path: &Path) {
        let file = File::create(path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        zip.start_file("lua/test.lua", zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(b"return 1").unwrap();
        zip.add_directory("lua/empty/", zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.finish().unwrap();
    }

    #[test]
    fn lists_entries_sorted() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("farust-test-{}.scd", std::process::id()));
        make_test_zip(&path);
        let entries = list_scd(&path).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries[0].name < entries[1].name);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn reads_inner_file() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("farust-test-read-{}.scd", std::process::id()));
        make_test_zip(&path);
        let data = read_scd_file(&path, "lua/test.lua").unwrap();
        assert_eq!(data, b"return 1");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn reads_inner_file_case_insensitive() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("farust-test-ci-{}.scd", std::process::id()));
        make_test_zip(&path);
        let (actual, data) = read_scd_file_ci(&path, "/LUA/TEST.LUA").unwrap();
        assert_eq!(actual, "lua/test.lua");
        assert_eq!(data, b"return 1");
        std::fs::remove_file(&path).ok();
    }
}
