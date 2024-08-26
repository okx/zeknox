
use bit_reverse:: ParallelReverse;

pub fn transpose<T: Send + Sync + Copy>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
    let len = matrix[0].len();
    (0..len)
        .into_iter()
        .map(|i| matrix.iter().map(|row| row[i]).collect())
        .collect()
}

pub fn transpose_and_rev<T: Send + Sync + Copy>(matrix: &[Vec<T>], lg_n: usize) -> Vec<Vec<T>> {
    let mut transposed = transpose(matrix);
    let length = transposed.len();
    for i in 0..length{
        let inversed = i.swap_bits() >> (64 - lg_n);
        if inversed > i {
            let inv = transposed[inversed].clone();
            let got = std::mem::replace(&mut transposed[i], inv);
            transposed[inversed] = got;

        }
    }  
    transposed
}