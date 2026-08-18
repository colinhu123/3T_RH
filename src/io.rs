use crate::state;

use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;

pub fn save_data(
    u: &crate::field1::Field,
    filename: &str,
    lx: f64,
    ly: f64,
) {
    use std::fs::{
        create_dir_all,
        File,
    };


    create_dir_all("data_new")
        .expect(
            "Cannot create data directory"
        );

    let path =
        format!("data/{}", filename);

    let mut file =
        File::create(path)
        .expect(
            "Cannot create output file"
        );

    writeln!(
        file,
        "x,y,rho,mom_x,mom_y,ee,ei,er"
    )
    .unwrap();

    let nx = u.grid.nx;
    let ny = u.grid.ny;

    let dx = lx / nx as f64;
    let dy = ly / ny as f64;

    for j in 0..ny {
        for i in 0..nx {
            let state =
                u.get((
                    i as isize,
                    j as isize,
                ));

            let x =
                (i as f64 + 0.5)
                * dx;

            let y =
                (j as f64 + 0.5)
                * dy;

            writeln!(
                file,
                "{},{},{},{},{},{},{},{}",
                x,
                y,
                state.rho,
                state.mom_x,
                state.mom_y,
                state.ee,
                state.ei,
                state.er,
            )
            .unwrap();
        }

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