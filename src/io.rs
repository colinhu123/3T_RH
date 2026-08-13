use crate::state;

use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;

type Grid = Vec<Vec<state::State>>;
pub fn save_data(
    u: &Grid,
    filename: &str,
    lx: f64,
    ly: f64,
) {
    create_dir_all("data")
        .expect("Cannot create data directory");

    let path =
        format!("data/{}", filename);

    let mut file =
        File::create(path)
        .expect("Cannot create output file");

    writeln!(
        file,
        "x,y,rho,mom_x,mom_y,ee,ei,er"
    )
    .unwrap();

    let nx = u.len();
    let ny = u[0].len();

    let dx = lx / nx as f64;
    let dy = ly / ny as f64;

    for j in 0..ny {
        for i in 0..nx {
            let x =
                (i as f64 + 0.5) * dx;

            let y =
                (j as f64 + 0.5) * dy;

            writeln!(
                file,
                "{},{},{},{},{},{},{},{}",
                x,
                y,
                u[i][j].rho,
                u[i][j].mom_x,
                u[i][j].mom_y,
                u[i][j].ee,
                u[i][j].ei,
                u[i][j].er,
            )
            .unwrap();
        }

        // Helpful for gnuplot structured-grid plots
        writeln!(file).unwrap();
    }
}


pub fn clear_data_folder() {

    let path = "data";

    if std::path::Path::new(path).exists() {

        remove_dir_all(path)
            .expect("Failed to remove old data folder");

    }

    create_dir_all(path)
        .expect("Failed to create data folder");
}