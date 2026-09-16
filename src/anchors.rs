//! Derived source anchors. Only the marker in a payload is durable; no snapshot field.
use crate::{ids::MemoryId, payload::MemoryPayload};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Component, Path};

pub const MARKER: &str = "\n[anchor] ";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Anchor {
    pub repo: String,
    pub path: String,
    pub scope: String,
    pub content_hash: String,
}

impl Anchor {
    pub fn parse(content: &[u8]) -> Option<Self> {
        let text = std::str::from_utf8(content).ok()?;
        if !(text.starts_with("[artifact]") || text.starts_with("[done]")) { return None; }
        let (_, marker) = text.rsplit_once(MARKER)?;
        let anchor: Self = serde_json::from_str(marker.trim()).ok()?;
        let relative = Path::new(&anchor.path);
        if !Path::new(&anchor.repo).is_absolute() || relative.is_absolute()
            || anchor.path.is_empty() || anchor.scope.is_empty()
            || relative.components().any(|c| !matches!(c, Component::Normal(_)))
            || anchor.content_hash.len() != 64
            || !anchor.content_hash.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        { return None; }
        Some(anchor)
    }

    pub fn same_source(&self, other: &Self) -> bool {
        self.repo == other.repo && self.path == other.path && self.scope == other.scope
    }
}

#[derive(Clone)]
pub struct Entry {
    pub anchor: Anchor,
    pub realm: String,
    pub created_at_ms: i64,
}

#[derive(Default)]
pub struct AnchorIndex {
    pub entries: HashMap<MemoryId, Entry>,
}

impl AnchorIndex {
    pub fn upsert(&mut self, id: MemoryId, payload: &MemoryPayload) {
        self.entries.remove(&id);
        if let Some(anchor) = Anchor::parse(&payload.content) {
            self.entries.insert(id, Entry { anchor, realm: payload.realm.clone(), created_at_ms: payload.created_at_ms });
        }
    }
    pub fn rebuild(payloads: &HashMap<MemoryId, MemoryPayload>) -> Self {
        let mut index = Self::default();
        for (&id, payload) in payloads { index.upsert(id, payload); }
        index
    }
}

impl crate::field::ChittaField {
    pub fn source_anchors(&self, request: &serde_json::Value) -> serde_json::Value {
        use sha2::{Digest, Sha256};
        use std::io::Read;
        let wanted: Option<Anchor> = request.get("anchor").and_then(|v| serde_json::from_value(v.clone()).ok());
        let ids = request.get("ids").and_then(|v| v.as_array());
        let realm = request.get("realm").and_then(|v| v.as_str()).unwrap_or("");
        let path = request.get("path").and_then(|v| v.as_str()).unwrap_or("");
        // Clone under short guards. Never hold a store lock during filesystem IO.
        let entries = self.anchors.read().entries.clone();
        let mut selected = Vec::new();
        {
            let states = self.states.read();
            for (id, entry) in entries {
                let Some(state) = states.get(&id).filter(|s| !s.deleted) else { continue };
                if !realm.is_empty() && entry.realm != realm { continue; }
                if let Some(ids) = ids { if !ids.iter().any(|v| v.as_str().and_then(|s| s.parse::<u64>().ok()) == Some(id)) { continue; } }
                if let Some(ref anchor) = wanted { if !anchor.same_source(&entry.anchor) { continue; } }
                if !path.is_empty() && Path::new(&entry.anchor.repo).join(&entry.anchor.path) != Path::new(path) { continue; }
                selected.push((id, entry, state.status.clone() as u8));
            }
        }
        selected.sort_by_key(|(id, e, _)| (e.created_at_ms, *id));
        let mut hashes: HashMap<std::path::PathBuf, Result<String, String>> = HashMap::new();
        serde_json::Value::Array(selected.into_iter().map(|(id, entry, status)| {
            let anchor = entry.anchor;
            let file = Path::new(&anchor.repo).join(&anchor.path);
            let current = hashes.entry(file.clone()).or_insert_with(|| {
                let canonical = file.canonicalize().map_err(|e| if e.kind() == std::io::ErrorKind::NotFound { "missing" } else { "unavailable" }.to_string())?;
                let root = Path::new(&anchor.repo).canonicalize().map_err(|_| "unavailable".to_string())?;
                if !canonical.starts_with(root) { return Err("unavailable".into()); }
                let mut reader = std::fs::File::open(canonical).map_err(|_| "unavailable".to_string())?;
                let mut digest = Sha256::new();
                let mut buf = [0u8; 16384];
                loop { let n = reader.read(&mut buf).map_err(|_| "unavailable".to_string())?; if n == 0 { break; } digest.update(&buf[..n]); }
                Ok(format!("{:x}", digest.finalize()))
            });
            let state = match current { Ok(hash) if *hash == anchor.content_hash => "current", Ok(_) => "stale", Err(s) => s.as_str() };
            serde_json::json!({"id":id.to_string(), "anchor":anchor, "state":state, "status":status})
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anchor_identity_and_validation() {
        let a = Anchor { repo: "/repo".into(), path: "docs/guide.md".into(), scope: "Runtime".into(), content_hash: "a".repeat(64) };
        let content = format!("[artifact] input:docs/guide.md{MARKER}{}", serde_json::to_string(&a).unwrap());
        assert_eq!(Anchor::parse(content.as_bytes()), Some(a.clone()));
        assert!(Anchor::parse(content.replace("[artifact]", "[wisdom]").as_bytes()).is_none());
        assert!(Anchor::parse(content.replace("docs/guide.md", "../secret").as_bytes()).is_none());
        let mut b = a.clone(); b.content_hash = "b".repeat(64);
        assert!(a.same_source(&b));
        b.scope = "Other heading".into(); assert!(!a.same_source(&b));
        b.scope = a.scope.clone(); b.repo = "/other".into(); assert!(!a.same_source(&b));
    }

    #[test]
    fn lifecycle_replay_edits_deletion_and_foreign_apply() {
        use sha2::{Digest, Sha256};
        use crate::field::ChittaField;
        let repo = tempfile::tempdir().unwrap();
        let store = tempfile::tempdir().unwrap();
        let path = repo.path().join("guide.md");
        std::fs::write(&path, "version one").unwrap();
        let anchor = Anchor { repo: repo.path().to_str().unwrap().into(), path: "guide.md".into(), scope: "Runtime".into(), content_hash: format!("{:x}", Sha256::digest(b"version one")) };
        let content = format!("[artifact] input:guide.md{MARKER}{}", serde_json::to_string(&anchor).unwrap());
        let request = serde_json::json!({"realm":"project:test"});
        let field = ChittaField::open(store.path().to_path_buf()).unwrap();
        let peer = ChittaField::open_unlocked(store.path().to_path_buf()).unwrap();
        let (id, _) = field.put_memory("signal", "project:test", content.as_bytes(), &[], 0.8, 0.0, 0, vec![], None, Some("hook".into())).unwrap();
        assert_eq!(field.source_anchors(&request)[0]["state"], "current");
        assert!(field.source_anchors(&serde_json::json!({"realm":"project:other"})).as_array().unwrap().is_empty());
        field.flush().unwrap();
        peer.sync_foreign().unwrap();
        assert_eq!(peer.source_anchors(&request)[0]["id"], id.to_string());
        drop(peer); drop(field);
        std::fs::write(&path, "version two").unwrap();
        let reopened = ChittaField::open(store.path().to_path_buf()).unwrap();
        assert_eq!(reopened.source_anchors(&request)[0]["state"], "stale");
        std::fs::remove_file(path).unwrap();
        assert_eq!(reopened.source_anchors(&request)[0]["state"], "missing");
        reopened.forget(id).unwrap();
        assert!(reopened.source_anchors(&request).as_array().unwrap().is_empty());
    }
}
