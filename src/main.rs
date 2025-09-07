use ndarray::{array, Array1, Array2, Axis, s};
use serde::Deserialize;
use std::error::Error;

#[derive(Debug, Clone)]
struct LogisticRegression {
    weights: Array1<f64>,
    bias: f64,
    lr: f64,
    l2: f64,
    epochs: usize,
}

impl LogisticRegression {
    fn new(n_features: usize, lr: f64, l2: f64, epochs: usize) -> Self {
        Self {
            weights: Array1::zeros(n_features),
            bias: 0.0,
            lr,
            l2,
            epochs,
        }
    }

    fn sigmoid(z: &Array1<f64>) -> Array1<f64> {
        // numerically-stable sigmoid
        z.map(|&x| {
            if x >= 0.0 {
                let e = (-x).exp();
                1.0 / (1.0 + e)
            } else {
                let e = x.exp();
                e / (1.0 + e)
            }
        })
    }

    fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) {
        let n_samples = X.nrows() as f64;

        for _ in 0..self.epochs {
            let z = X.dot(&self.weights) + self.bias;
            let y_hat = Self::sigmoid(&z);

            let error = &y_hat - y; 
            let grad_w = X.t().dot(&error) / n_samples + (&self.weights * self.l2) / n_samples;
            let grad_b = error.sum() / n_samples;

            self.weights -= &(grad_w * self.lr);
            self.bias -= self.lr * grad_b;
        }
    }

    fn predict_proba(&self, X: &Array2<f64>) -> Array1<f64> {
        let z = X.dot(&self.weights) + self.bias;
        Self::sigmoid(&z)
    }

    fn predict(&self, X: &Array2<f64>, threshold: f64) -> Array1<u8> {
        self.predict_proba(X)
            .map(|p| if *p >= threshold { 1 } else { 0 })
    }
}

// Akurasi
fn accuracy(y_true: &Array1<u8>, y_pred: &Array1<u8>) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| a == b)
        .count();
    correct as f64 / y_true.len() as f64
}

// Struktur CSV row
#[derive(Debug, Deserialize)]
struct Row {
    age: f64,
    hypertension: f64,
    heart_disease: f64,
    avg_glucose_level: f64,
    bmi: f64,
    stroke: u8,
}

// Loader CSV -> ndarray
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

fn main() -> Result<(), Box<dyn Error>> {
    // 1) load data
    let (X, y, y_label) = load_csv("./BMI_Stroke.csv")?;
    println!("Loaded dataset with {} samples", X.nrows());

    // 2) standardisasi
    let mean = X.mean_axis(Axis(0)).unwrap();
    let std = X.std_axis(Axis(0), 0.0);
    let X_std = X
        .rows()
        .into_iter()
        .map(|row| &row.to_owned() - &mean)
        .map(|row| &row / &(&std + 1e-9))
        .collect::<Vec<_>>();
    let X_std = Array2::from_shape_vec((X.nrows(), X.ncols()), X_std.into_iter().flatten().collect()).unwrap();

    // 3) split train/test
    let n_train = (X_std.nrows() as f64 * 0.8) as usize;
    let X_train = X_std.slice(s![0..n_train, ..]).to_owned();
    let y_train = y.slice(s![0..n_train]).to_owned();
    let y_train_u8 = y_label.slice(s![0..n_train]).to_owned();

    let X_test = X_std.slice(s![n_train.., ..]).to_owned();
    let y_test_u8 = y_label.slice(s![n_train..]).to_owned();

    // 4) train
    let mut model = LogisticRegression::new(5, 0.1, 0.01, 2000);
    model.fit(&X_train, &y_train);

    // 5) evaluasi
    let y_pred = model.predict(&X_test, 0.5);
    let acc = accuracy(&y_test_u8, &y_pred);

    println!("Weights  : {:?}", model.weights);
    println!("Bias     : {:?}", model.bias);
    println!("Accuracy : {:.4}", acc);

    Ok(())
}
