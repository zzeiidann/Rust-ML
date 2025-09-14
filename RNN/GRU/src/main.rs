use ndarray::{Array1, Array2, Array3, Axis, s};
use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use rand::Rng;

// ===== Utilities =====
fn mse_loss(y_pred: &Array1<f64>, y_true: &Array1<f64>) -> f64 {
    let diff = y_pred - y_true;
    (diff.mapv(|v| v * v).sum()) / (y_true.len() as f64)
}

// ===== GRU Model (forward only + output layer trainable) =====
#[derive(Debug, Clone)]
struct GRU {
    // GRU gate weights
    w_z: Array2<f64>, u_z: Array2<f64>, b_z: Array1<f64>, // update gate
    w_r: Array2<f64>, u_r: Array2<f64>, b_r: Array1<f64>, // reset gate
    w_h: Array2<f64>, u_h: Array2<f64>, b_h: Array1<f64>, // candidate hidden

    // output layer (trainable)
    w_hy: Array2<f64>, // hidden_size x 1
    b_y: f64,

    input_size: usize,
    hidden_size: usize,
    lr: f64,
}

impl GRU {
    fn new(input_size: usize, hidden_size: usize, lr: f64) -> Self {
        let mut rng = rand::thread_rng();

        let mut init = |r1: usize, r2: usize| {
            Array2::from_shape_fn((r1, r2), |_| rng.gen_range(-0.08..0.08))
        };

        let w_z = init(input_size, hidden_size);
        let u_z = init(hidden_size, hidden_size);
        let b_z = Array1::zeros(hidden_size);

        let w_r = init(input_size, hidden_size);
        let u_r = init(hidden_size, hidden_size);
        let b_r = Array1::zeros(hidden_size);

        let w_h = init(input_size, hidden_size);
        let u_h = init(hidden_size, hidden_size);
        let b_h = Array1::zeros(hidden_size);

        let w_hy = init(hidden_size, 1);
        let b_y = 0.0;

        Self {
            w_z, u_z, b_z,
            w_r, u_r, b_r,
            w_h, u_h, b_h,
            w_hy, b_y,
            input_size,
            hidden_size,
            lr,
        }
    }

    // Activation helpers
    fn sigmoid(a: &Array2<f64>) -> Array2<f64> {
        a.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }
    fn tanh(a: &Array2<f64>) -> Array2<f64> {
        a.mapv(|x| x.tanh())
    }

    // Forward pass
    fn forward(&self, X: &Array3<f64>) -> (Array2<f64>, Vec<Array2<f64>>) {
        let batch = X.dim().0;
        let seq_len = X.dim().1;

        let mut h_t = Array2::<f64>::zeros((batch, self.hidden_size));
        let mut hs: Vec<Array2<f64>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = X.slice(s![.., t, ..]).to_owned(); // (batch, input_size)

            let z_t = Self::sigmoid(&(x_t.dot(&self.w_z) + h_t.dot(&self.u_z) + &self.b_z.view().insert_axis(Axis(0))));
            let r_t = Self::sigmoid(&(x_t.dot(&self.w_r) + h_t.dot(&self.u_r) + &self.b_r.view().insert_axis(Axis(0))));

            let h_tilde = Self::tanh(&(x_t.dot(&self.w_h) + (r_t * &h_t).dot(&self.u_h) + &self.b_h.view().insert_axis(Axis(0))));

            h_t = (1.0 - &z_t) * h_t + &z_t * h_tilde;
            hs.push(h_t.clone());
        }

        (h_t, hs)
    }

    // Predict
    fn predict(&self, X: &Array3<f64>) -> Array1<f64> {
        let (h_last, _hs) = self.forward(X);
        let y_logits = h_last.dot(&self.w_hy) + self.b_y;
        y_logits.column(0).to_owned()
    }

    // Train output layer only
    fn fit_output_only(&mut self, X: &Array3<f64>, y: &Array1<f64>, epochs: usize) {
        let n = X.dim().0 as f64;
        for epoch in 0..epochs {
            let (h_last, _hs) = self.forward(X);
            let y_pred = h_last.dot(&self.w_hy) + self.b_y; // shape (batch,1)
            let y_pred_col = y_pred.column(0).to_owned();

            let diff = &y_pred_col - y;
            let grad_w = h_last.t().dot(&diff.view().insert_axis(Axis(1))).mapv(|v| v / n);
            let grad_b = diff.sum() / n;

            self.w_hy = &self.w_hy - &(grad_w * self.lr);
            self.b_y -= self.lr * grad_b;

            if epoch % 100 == 0 {
                let loss = mse_loss(&y_pred_col, y);
                println!("Epoch {}: MSE={:.6}", epoch, loss);
            }
        }
    }
}

// ===== Data Loading & Processing =====
fn load_series(path: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let mut series: Vec<f64> = Vec::new();
    for result in rdr.records() {
        let rec = result?;
        let close_str = rec.get(1).ok_or("Missing Close column")?;
        let v: f64 = close_str.parse()?;
        series.push(v);
    }
    Ok(series)
}

fn make_dataset(series: &[f64], seq_len: usize) -> (Array3<f64>, Array1<f64>) {
    let n_samples = series.len().saturating_sub(seq_len);
    let mut X = Array3::<f64>::zeros((n_samples, seq_len, 1));
    let mut y = Array1::<f64>::zeros(n_samples);

    for i in 0..n_samples {
        for t in 0..seq_len {
            X[[i, t, 0]] = series[i + t];
        }
        y[i] = series[i + seq_len];
    }

    let all_values: Vec<f64> = X.iter().cloned().collect();
    let mean: f64 = all_values.iter().sum::<f64>() / all_values.len() as f64;
    let std: f64 = (all_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / all_values.len() as f64).sqrt();
    let std = if std == 0.0 { 1e-8 } else { std };

    let mut X_norm = X.clone();
    for v in X_norm.iter_mut() {
        *v = (*v - mean) / std;
    }
    let mut y_norm = y.clone();
    for vi in y_norm.iter_mut() {
        *vi = (*vi - mean) / std;
    }

    (X_norm, y_norm)
}

fn save_predictions(path: &str, timestamps: &[String], y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<(), Box<dyn Error>> {
    let mut w = File::create(path)?;
    writeln!(w, "Timestamp,True,Pred")?;
    for i in 0..y_true.len() {
        writeln!(w, "{},{:.6},{:.6}", timestamps[i], y_true[i], y_pred[i])?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== GRU (output-only training) for BTC hourly regression ===");

    let seq_len = 24;
    let hidden_size = 64;
    let lr = 0.001;
    let epochs = 1000;

    let series = load_series("btc_close_hourly.csv")?;
    if series.len() <= seq_len + 1 {
        return Err("Series too short".into());
    }

    let (X_all, y_all) = make_dataset(&series, seq_len);

    let n_samples = X_all.dim().0;
    let n_test = (n_samples as f64 * 0.1).ceil() as usize;
    let n_train = n_samples - n_test;

    let X_train = X_all.slice(s![0..n_train, .., ..]).to_owned();
    let y_train = y_all.slice(s![0..n_train]).to_owned();

    let X_test = X_all.slice(s![n_train.., .., ..]).to_owned();
    let y_test = y_all.slice(s![n_train..]).to_owned();

    println!("Samples: total={}, train={}, test={}", n_samples, n_train, n_test);

    let mut model = GRU::new(1, hidden_size, lr);
    model.fit_output_only(&X_train, &y_train, epochs);

    let y_pred_train = model.predict(&X_train);
    let y_pred_test = model.predict(&X_test);

    println!("Train MSE: {:.6}", mse_loss(&y_pred_train, &y_train));
    println!("Test  MSE: {:.6}", mse_loss(&y_pred_test, &y_test));

    let timestamps: Vec<String> = (0..y_test.len()).map(|i| format!("idx_{}", i)).collect();
    save_predictions("predictions.csv", &timestamps, &y_test, &y_pred_test)?;
    println!("Predictions saved to predictions.csv");

    Ok(())
}
