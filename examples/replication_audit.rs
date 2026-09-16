//! Usage: replication_audit PRIVATE_COPY/chitta-field
//! ChittaField::open is exclusive and may write caches; never use a live store.
fn main() {
    let path = std::env::args().nth(1).expect("explicit private store copy required");
    let field = chitta_field::ChittaField::open(path.into()).expect("open private copy");
    println!("{}", field.replication_audit());
}
