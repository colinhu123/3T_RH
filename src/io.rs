use std::fs::{create_dir_all, remove_dir_all, File, rename};
use std::io::{BufWriter, Write};
use std::path::Path;

const MAGIC: [u8; 8] = *b"RH3TBIN1";
const VERSION: u32 = 1;
const NVAR: u32 = 8;

/// Binary layout (little endian):
///
/// header:
///   [8]u8  magic = "RH3TBIN1"
///   u32    version
///   u32    nvar (=8)
///   u64    nx
///   u64    ny
///   f64    time
///
/// payload, j-major / i-fastest, exactly as the old text writer:
///   repeated nx*ny times:
///   f64 x, y, rho, mom_x, mom_y, ee, ei, er
///
/// The file is first written as *.tmp and atomically renamed to the requested
/// final name, so a live Python visualizer never sees a partially-written file.
pub fn save_data(
    u: &crate::field1::Field,
    filename: &str,
    _lx: f64,
    _ly: f64,
) {
    create_dir_all("data_new").expect("Cannot create data directory");

    let final_path = format!("data_new/{}", filename);
    let tmp_path = format!("{}.tmp", final_path);

    let file = File::create(&tmp_path).expect("Cannot create temporary output file");
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);

    let nx = u.grid.nx;
    let ny = u.grid.ny;

    writer.write_all(&MAGIC).unwrap();
    writer.write_all(&VERSION.to_le_bytes()).unwrap();
    writer.write_all(&NVAR.to_le_bytes()).unwrap();
    writer.write_all(&(nx as u64).to_le_bytes()).unwrap();
    writer.write_all(&(ny as u64).to_le_bytes()).unwrap();
    writer.write_all(&u.time.to_le_bytes()).unwrap();

    // Keep the old ordering: j outer, i inner, so x changes fastest.
    for j in 0..ny {
        for i in 0..nx {
            let ii = i as isize;
            let jj = j as isize;

            // Do NOT call Field::get for non-fluid Cartesian points:
            // for polygon/disk cases that would trigger an expensive BC solve.
            // Store NaNs outside the physical domain so Python can mask them.
            let (rho, mom_x, mom_y, ee, ei, er) =
                if u.is_in_domain((ii, jj)) {
                    let s = u.value[u.linear_index((ii, jj))];
                    (s.rho, s.mom_x, s.mom_y, s.ee, s.ei, s.er)
                } else {
                    let z = f64::NAN;
                    (z, z, z, z, z, z)
                };

            // Use GridInfo's actual coordinates. This is important for domains
            // whose x0/y0 are not zero (e.g. DMR and disk benchmarks).
            let x = u.grid.x(ii);
            let y = u.grid.y(jj);

            for value in [x, y, rho, mom_x, mom_y, ee, ei, er] {
                writer.write_all(&value.to_le_bytes()).unwrap();
            }
        }
    }

    writer.flush().expect("Failed to flush output file");
    drop(writer);

    // On Unix/WSL, rename within one directory is atomic.
    // Remove an existing target first only if necessary for portability.
    if Path::new(&final_path).exists() {
        std::fs::remove_file(&final_path).expect("Failed to replace old output file");
    }
    rename(&tmp_path, &final_path).expect("Failed to publish completed output file");
}

pub fn clear_data_folder() {
    let path = "data_new";

    if Path::new(path).exists() {
        remove_dir_all(path).expect("Failed to remove old data folder");
    }

    create_dir_all(path).expect("Failed to create data folder");
}

pub fn load_data(u: &mut crate::field1::Field, path: &str) {
    use std::io::{BufReader, Read};
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open restart file {}: {}", path, e));
    let mut r = BufReader::with_capacity(16 * 1024 * 1024, file);
    let mut magic=[0u8;8]; r.read_exact(&mut magic).unwrap(); assert_eq!(magic, MAGIC);
    let mut b4=[0u8;4]; let mut b8=[0u8;8];
    r.read_exact(&mut b4).unwrap(); assert_eq!(u32::from_le_bytes(b4), VERSION);
    r.read_exact(&mut b4).unwrap(); assert_eq!(u32::from_le_bytes(b4), NVAR);
    r.read_exact(&mut b8).unwrap(); let nx=u64::from_le_bytes(b8) as usize;
    r.read_exact(&mut b8).unwrap(); let ny=u64::from_le_bytes(b8) as usize;
    r.read_exact(&mut b8).unwrap(); let time=f64::from_le_bytes(b8);
    assert_eq!((nx,ny),(u.grid.nx,u.grid.ny),"restart grid mismatch");
    for j in 0..ny { for i in 0..nx {
        let mut a=[0.0f64;8];
        for q in 0..8 { r.read_exact(&mut b8).unwrap(); a[q]=f64::from_le_bytes(b8); }
        let idx=(i as isize,j as isize);
        if u.is_in_domain(idx) {
            let s=crate::state::State{rho:a[2],mom_x:a[3],mom_y:a[4],ee:a[5],ei:a[6],er:a[7]};
            assert!(s.rho.is_finite()&&s.mom_x.is_finite()&&s.mom_y.is_finite()&&s.ee.is_finite()&&s.ei.is_finite()&&s.er.is_finite(),
                    "non-finite restart state at {:?}",idx);
            let k=u.linear_index(idx); u.value[k]=s;
        }
    }}
    u.time=time;
    println!("Restart loaded: {} at t={:.16e}",path,time);
}