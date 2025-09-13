use ndarray::{array, Array1, Array2, Axis, s};
use serde::Deserialize;
use std::error::Error;
use rand::Rng;
use std::time::Instant;

// ===== RNN Model =====
#[derive(Debug, Clone)]
struct RNN {
    w_xh: Array2<f64>, // input → hidden
    w_hh: Array2<f64>, // hidden → hidden
    b_h: Array1<f64>,  // bias hidden
    w_hy: Array2<f64>, // hidden → output
    b_y: f64,          // bias output
    hidden_size: usize,
    lr: f64,
    epochs: usize,
}

impl RNN {
    fn new(input_size: usize, hidden_size: usize, lr: f64, epochs: usize) -> Self {
        let mut rng = rand::thread_rng();

        let limit_xh = (6.0 / (input_size + hidden_size) as f64).sqrt();
        let limit_hh = (6.0 / (hidden_size + hidden_size) as f64).sqrt();
        let limit_hy = (6.0 / (hidden_size + 1) as f64).sqrt();

        let w_xh = Array2::from_shape_fn((input_size, hidden_size), |_| rng.gen_range(-limit_xh..limit_xh));
        let w_hh = Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.gen_range(-limit_hh..limit_hh));
        let b_h = Array1::zeros(hidden_size);

        let w_hy = Array2::from_shape_fn((hidden_size, 1), |_| rng.gen_range(-limit_hy..limit_hy));
        let b_y = 0.0;

        Self {
            w_xh,
            w_hh,
            b_h,
            w_hy,
            b_y,
            hidden_size,
            lr,
            epochs,
        }
    }

    fn tanh(x: &Array2<f64>) -> Array2<f64> {
        x.map(|&v| v.tanh())
    }

    fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
        x.map(|&v| 1.0 / (1.0 + (-v).exp()))
    }

    // Forward pass per batch (treat each row as sequence of features)
    fn forward(&self, X: &Array2<f64>) -> (Vec<Array2<f64>>, Array2<f64>) {
        let n_samples = X.nrows();
        let input_len = X.ncols();

        let mut hs: Vec<Array2<f64>> = Vec::new();
        let mut h_prev = Array2::zeros((n_samples, self.hidden_size));

        // process each feature as timestep
        for t in 0..input_len {
            let x_t = X.slice(s![.., t..t+1]); // ambil fitur ke-t
            let h_t = Self::tanh(&(x_t.dot(&self.w_xh) + h_prev.dot(&self.w_hh) + &self.b_h));
            h_prev = h_t.clone();
            hs.push(h_t);
        }

        let y_logits = h_prev.dot(&self.w_hy) + self.b_y;
        let y_pred = Self::sigmoid(&y_logits);
        (hs, y_pred)
    }

    fn compute_loss(y_pred: &Array2<f64>, y_true: &Array2<f64>) -> f64 {
        let eps = 1e-15;
        let y_pred = y_pred.map(|&p| p.max(eps).min(1.0 - eps));
        let pos = y_true * y_pred.map(|v| v.ln());
        let neg = (1.0 - y_true) * (1.0 - &y_pred).map(|v| v.ln());
        -(pos + neg).mean().unwrap()
    }

    fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) {
        let n_samples = X.nrows() as f64;
        let y_col = y.clone().into_shape((y.len(), 1)).unwrap();

        for epoch in 0..self.epochs {
            let (_hs, y_pred) = self.forward(X);

            // gradient (output layer)
            let dz = &y_pred - &y_col;
            let dw_hy = _hs.last().unwrap().t().dot(&dz) / n_samples;
            let db_y = dz.sum() / n_samples;

            // update hanya output layer (sederhana)
            self.w_hy -= &(dw_hy * self.lr);
            self.b_y -= db_y * self.lr;

            if epoch % 200 == 0 {
                let loss = Self::compute_loss(&y_pred, &y_col);
                println!("Epoch {}: Loss = {:.6}", epoch, loss);
            }
        }
    }

    fn predict_proba(&self, X: &Array2<f64>) -> Array1<f64> {
        let (_, y_pred) = self.forward(X);
        y_pred.column(0).to_owned()
    }

    fn predict(&self, X: &Array2<f64>, threshold: f64) -> Array1<u8> {
        self.predict_proba(X).map(|&p| if p >= threshold { 1 } else { 0 })
    }
}

// ===== Metrics =====
fn accuracy(y_true: &Array1<u8>, y_pred: &Array1<u8>) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| a == b)
        .count();
    correct as f64 / y_true.len() as f64
}

fn precision(y_true: &Array1<u8>, y_pred: &Array1<u8>) -> f64 {
    let tp = y_true.iter().zip(y_pred.iter())
        .filter(|(&t, &p)| t == 1 && p == 1).count() as f64;
    let fp = y_true.iter().zip(y_pred.iter())
        .filter(|(&t, &p)| t == 0 && p == 1).count() as f64;
    if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) }
}

fn recall(y_true: &Array1<u8>, y_pred: &Array1<u8>) -> f64 {
    let tp = y_true.iter().zip(y_pred.iter())
        .filter(|(&t, &p)| t == 1 && p == 1).count() as f64;
    let fn_count = y_true.iter().zip(y_pred.iter())
        .filter(|(&t, &p)| t == 1 && p == 0).count() as f64;
    if tp + fn_count == 0.0 { 0.0 } else { tp / (tp + fn_count) }
}

fn f1_score(precision: f64, recall: f64) -> f64 {
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * (precision * recall) / (precision + recall)
    }
}

// ===== CSV Row =====
#[derive(Debug, Deserialize)]
struct Row {
    age: f64,
    hypertension: f64,
    heart_disease: f64,
    avg_glucose_level: f64,
    bmi: f64,
    stroke: u8,
}

// ===== CSV Loader =====
fn load_csv(path: &str) -> Result<(Array2<f64>, Array1<f64>, Array1<u8>), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut features: Vec<f64> = Vec::new();
    let mut labels_f: Vec<f64> = Vec::new();
    let mut labels_u8: Vec<u8> = Vec::new();

    for result in rdr.deserialize() {
        let row: Row = result?;
        features.extend_from_slice(&[
            row.age,
            row.hypertension,
            row.heart_disease,
            row.avg_glucose_level,
            row.bmi,
        ]);
        labels_f.push(row.stroke as f64);
        labels_u8.push(row.stroke);
    }

    let n_samples = labels_f.len();
    let X = Array2::from_shape_vec((n_samples, 5), features)?;
    let y = Array1::from(labels_f);
    let y_u8 = Array1::from(labels_u8);

    Ok((X, y, y_u8))
}

// ===== MAIN =====
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== RNN for Stroke Prediction ===\n");

    // 1) Load data
    let (X, y, y_label) = load_csv("./BMI_Stroke.csv")?;
    println!("Loaded dataset with {} samples and {} features", X.nrows(), X.ncols());

    // 2) Standardization
    let mean = X.mean_axis(Axis(0)).unwrap();
    let std = X.std_axis(Axis(0), 0.0);
    let X_std = X
        .rows()
        .into_iter()
        .map(|row| &row.to_owned() - &mean)
        .map(|row| &row / &(&std + 1e-9))
        .collect::<Vec<_>>();
    let X_std = Array2::from_shape_vec((X.nrows(), X.ncols()), X_std.into_iter().flatten().collect()).unwrap();

    // 3) Train/test split
    let n_train = (X_std.nrows() as f64 * 0.8) as usize;
    let X_train = X_std.slice(s![0..n_train, ..]).to_owned();
    let y_train = y.slice(s![0..n_train]).to_owned();
    let y_train_u8 = y_label.slice(s![0..n_train]).to_owned();

    let X_test = X_std.slice(s![n_train.., ..]).to_owned();
    let y_test_u8 = y_label.slice(s![n_train..]).to_owned();

    println!("Train set: {} samples", X_train.nrows());
    println!("Test set:  {} samples\n", X_test.nrows());

    // 4) Create and train RNN
    let mut model = RNN::new(
        1,      // input size per timestep
        32,     // hidden size
        0.01,   // learning rate
        1800    // epochs
    );

    println!("Training RNN...");
    println!("Architecture: sequence(5 timesteps, 1 feature each) -> 10 hidden (tanh) -> 1 sigmoid\n");

    let start = Instant::now();
    model.fit(&X_train, &y_train);
    let duration = start.elapsed();
    println!("Training finished in {:.2?}", duration);

    // 5) Evaluation
    println!("\n=== Model Evaluation ===");
    let y_pred = model.predict(&X_test, 0.5);
    let y_pred_proba = model.predict_proba(&X_test);

    let acc = accuracy(&y_test_u8, &y_pred);
    let prec = precision(&y_test_u8, &y_pred);
    let rec = recall(&y_test_u8, &y_pred);
    let f1 = f1_score(prec, rec);

    println!("Accuracy:  {:.4}", acc);
    println!("Precision: {:.4}", prec);
    println!("Recall:    {:.4}", rec);
    println!("F1-Score:  {:.4}", f1);

    // 6) Sample predictions
    println!("\n=== Sample Predictions ===");
    for i in 0..5.min(X_test.nrows()) {
        println!(
            "Sample {}: True={}, Pred={}, Prob={:.4}",
            i + 1,
            y_test_u8[i],
            y_pred[i],
            y_pred_proba[i]
        );
    }

    // 7) Class distribution
    let positive_samples = y_test_u8.iter().filter(|&&x| x == 1).count();
    let negative_samples = y_test_u8.len() - positive_samples;
    println!("\n=== Test Set Distribution ===");
    println!("Negative (No Stroke): {} samples", negative_samples);
    println!("Positive (Stroke):    {} samples", positive_samples);

    println!("\nExecution Time: {:.2?}", duration);

    Ok(())
}
