use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type CodeFileId = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFile {
    pub id: CodeFileId,
    pub path: String,
    pub project: String,
    pub mtime: i64,
    #[serde(skip)]
    pub content_hash: Option<String>,
    #[serde(skip)]
    pub git_commit: Option<String>,
    #[serde(skip)]
    pub git_author: Option<String>,
    #[serde(skip)]
    pub git_timestamp_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFileIndex {
    by_id: HashMap<CodeFileId, CodeFile>,
    by_path: HashMap<String, CodeFileId>,
}

impl CodeFileIndex {
    pub fn new() -> Self {
        Self {
            by_id: HashMap::new(),
            by_path: HashMap::new(),
        }
    }

    /// Upsert a file. Returns (id, was_updated).
    /// `was_updated` is true when content_hash changed (or was absent before).
    pub fn upsert(
        &mut self,
        path: &str,
        project: &str,
        mtime: i64,
        content_hash: Option<String>,
        git_commit: Option<String>,
        git_author: Option<String>,
        git_timestamp_ms: Option<i64>,
        next_id: impl FnOnce() -> CodeFileId,
    ) -> (CodeFileId, bool) {
        if let Some(&existing_id) = self.by_path.get(path) {
            let entry = self.by_id.get_mut(&existing_id).unwrap();
            let hash_changed = match (&entry.content_hash, &content_hash) {
                (Some(old), Some(new)) => old != new,
                (None, Some(_)) => true,
                (Some(_), None) => true,
                (None, None) => true,
            };
            entry.mtime = mtime;
            entry.project = project.to_string();
            entry.content_hash = content_hash;
            entry.git_commit = git_commit;
            entry.git_author = git_author;
            entry.git_timestamp_ms = git_timestamp_ms;
            return (existing_id, hash_changed);
        }

        let id = next_id();
        let file = CodeFile {
            id,
            path: path.to_string(),
            project: project.to_string(),
            mtime,
            content_hash,
            git_commit,
            git_author,
            git_timestamp_ms,
        };
        self.by_path.insert(path.to_string(), id);
        self.by_id.insert(id, file);
        (id, true)
    }

    /// Get the stored content hash for a path.
    pub fn get_hash(&self, path: &str) -> Option<&str> {
        let &id = self.by_path.get(path)?;
        self.by_id.get(&id)?.content_hash.as_deref()
    }

    pub fn get_by_path(&self, path: &str) -> Option<&CodeFile> {
        let &id = self.by_path.get(path)?;
        self.by_id.get(&id)
    }

    pub fn get_by_project(&self, project: &str) -> Vec<&CodeFile> {
        let project = project.strip_prefix("project:").unwrap_or(project);
        let mut files: Vec<_> = self.by_id
            .values()
            .filter(|f| f.project.strip_prefix("project:").unwrap_or(&f.project) == project)
            .collect();
        files.sort_by(|a, b| a.path.cmp(&b.path));
        files
    }

    pub fn remove(&mut self, id: CodeFileId) {
        if let Some(file) = self.by_id.remove(&id) {
            self.by_path.remove(&file.path);
        }
    }

    pub fn count(&self) -> usize {
        self.by_id.len()
    }

    /// Return the maximum code file ID present, or None if empty.
    pub fn max_id(&self) -> Option<u64> {
        self.by_id.keys().copied().max()
    }

    /// Iterate all code files.
    pub fn iter(&self) -> impl Iterator<Item = &CodeFile> {
        self.by_id.values()
    }

    /// Remove all code files belonging to a project. Returns removed file paths.
    pub fn remove_by_project(&mut self, project: &str) -> Vec<String> {
        let project = project.strip_prefix("project:").unwrap_or(project);
        let ids_to_remove: Vec<CodeFileId> = self
            .by_id
            .iter()
            .filter(|(_, f)| f.project.strip_prefix("project:").unwrap_or(&f.project) == project)
            .map(|(&id, _)| id)
            .collect();
        let mut paths = Vec::new();
        for id in ids_to_remove {
            if let Some(file) = self.by_id.remove(&id) {
                self.by_path.remove(&file.path);
                paths.push(file.path);
            }
        }
        paths
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_aliases_share_sorted_coverage_and_removal() {
        let mut index = CodeFileIndex::new();
        index.upsert("/repo/z.rs", "project:demo", 0, None, None, None, None, || 1);
        index.upsert("/repo/a.rs", "demo", 0, None, None, None, None, || 2);
        index.upsert("/other/x.rs", "other", 0, None, None, None, None, || 3);
        for alias in ["demo", "project:demo"] {
            let files = index.get_by_project(alias);
            assert_eq!(files.len(), 2);
            assert_eq!(files[0].path, "/repo/a.rs");
        }
        assert_eq!(index.remove_by_project("project:demo").len(), 2);
        assert_eq!(index.count(), 1);
    }
}
