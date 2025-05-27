use functions::filter_intersecting::BBox;

mod functions;

fn main() {
    env_logger::init();
    // pollster::block_on(functions::map::run());
    let left = vec![
        BBox {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 2.0,
            max_y: 2.0,
        },
        BBox {
            min_x: 3.0,
            min_y: 3.0,
            max_x: 5.0,
            max_y: 5.0,
        },
    ];

    let right = vec![
        BBox {
            min_x: 1.0,
            min_y: 1.0,
            max_x: 4.0,
            max_y: 4.0,
        },
        BBox {
            min_x: 6.0,
            min_y: 6.0,
            max_x: 8.0,
            max_y: 8.0,
        },
    ];

    pollster::block_on(functions::filter_intersecting::filter_intersecting(left, right));
}
