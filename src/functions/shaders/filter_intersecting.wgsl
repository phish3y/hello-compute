struct BBox {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
};

struct Pair {
    left_index: u32,
    right_index: u32,
};

struct Counter {
    count: atomic<u32>,
};

@group(0) @binding(0)
var<storage, read> left_set: array<BBox>;

@group(0) @binding(1)
var<storage, read> right_set: array<BBox>;

@group(0) @binding(2)
var<storage, read_write> result: array<Pair>;

@group(0) @binding(3)
var<storage, read_write> counter: Counter;

fn intersects(a: BBox, b: BBox) -> bool {
    return !(a.max_x < b.min_x || a.min_x > b.max_x ||
             a.max_y < b.min_y || a.min_y > b.max_y);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let global_idx = global_id.x;

    let left_len = arrayLength(&left_set);
    let right_len = arrayLength(&right_set);
    let total_pairs = left_len * right_len;

    if (global_idx >= total_pairs) {
        return;
    }

    let i = global_idx / right_len;
    let j = global_idx % right_len;

    let a = left_set[i];
    let b = right_set[j];

    if (intersects(a, b)) {
        let idx = atomicAdd(&counter.count, 1u);
        result[idx] = Pair(i, j);
    }
}
