use ndarray::{array, Array1, Array2, Axis, s};
use serde::Deserialize;
use std::error::Error;
use rand::Rng;

#[derive(Debug, Clone)]
struct NeuralNetwork {
    // Layer weights and biases
    w1: Array2<f64>, // input to hidden
    b1: Array1<f64>,
    w2: Array2<f64>, // hidden to output
    b2: f64,
    lr: f64,
    epochs: usize,
    hidden_size: usize,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, lr: f64, epochs: usize) -> Self {
        let mut rng = rand::thread_rng();
        
        // Xavier initialization
        let limit1 = (6.0 / (input_size + hidden_size) as f64).sqrt();
        let limit2 = (6.0 / (hidden_size + 1) as f64).sqrt();
        
        let w1 = Array2::from_shape_fn((input_size, hidden_size), |_| {
            rng.gen_range(-limit1..limit1)
        });
        let b1 = Array1::zeros(hidden_size);
        
        let w2 = Array2::from_shape_fn((hidden_size, 1), |_| {
            rng.gen_range(-limit2..limit2)
        });
        let b2 = 0.0;

        Self {
            w1,
            b1,
            w2,
            b2,
            lr,
            epochs,
            hidden_size,
        }
    }

    // ReLU activation
    fn relu(x: &Array2<f64>) -> Array2<f64> {
        x.map(|&val| if val > 0.0 { val } else { 0.0 })
    }

    fn relu_derivative(x: &Array2<f64>) -> Array2<f64> {
        x.map(|&val| if val > 0.0 { 1.0 } else { 0.0 })
    }

    // Sigmoid activation
    fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
        x.map(|&val| {
            if val >= 0.0 {
                let e = (-val).exp();
                1.0 / (1.0 + e)
            } else {
                let e = val.exp();
                e / (1.0 + e)
            }
        })
    }

    fn sigmoid_derivative(x: &Array2<f64>) -> Array2<f64> {
        let sig = Self::sigmoid(x);
        &sig * &(1.0 - &sig)
    }

    // Forward pass
    fn forward(&self, X: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        // Hidden layer
        let z1 = X.dot(&self.w1) + &self.b1;
        let a1 = Self::relu(&z1);
        
        // Output layer
        let z2 = a1.dot(&self.w2) + self.b2;
        let a2 = Self::sigmoid(&z2);
        
        (a1, z2, a2)
    }

    // Training
    fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) {
        let n_samples = X.nrows() as f64;
        
        for epoch in 0..self.epochs {
            // Forward pass
            let (a1, z2, a2) = self.forward(X);
            
            // Convert y to column vector for broadcasting
            let y_col = y.clone().into_shape((y.len(), 1)).unwrap();
            
            // Backward pass
            // Output layer gradients
            let dz2 = &a2 - &y_col;
            let dw2 = a1.t().dot(&dz2) / n_samples;
            let db2 = dz2.sum() / n_samples;
            
            // Hidden layer gradients
            let da1 = dz2.dot(&self.w2.t());
            let z1 = X.dot(&self.w1) + &self.b1;
            let dz1 = &da1 * &Self::relu_derivative(&z1);
            let dw1 = X.t().dot(&dz1) / n_samples;
            let db1 = dz1.mean_axis(Axis(0)).unwrap();
            
            // Update weights and biases
            self.w1 -= &(dw1 * self.lr);
            self.b1 -= &(db1 * self.lr);
            self.w2 -= &(dw2 * self.lr);
            self.b2 -= db2 * self.lr;
            
            // Print loss every 200 epochs
            if epoch % 200 == 0 {
                let loss = self.compute_loss(&a2, &y_col);
                println!("Epoch {}: Loss = {:.6}", epoch, loss);
            }
        }
    }

    // Binary cross-entropy loss
    fn compute_loss(&self, y_pred: &Array2<f64>, y_true: &Array2<f64>) -> f64 {
        let epsilon = 1e-15;
        let y_pred_clipped = y_pred.map(|&p| p.max(epsilon).min(1.0 - epsilon));
        
        let pos_loss = y_true * y_pred_clipped.map(|p| p.ln());
        let neg_loss = (1.0 - y_true) * (1.0 - &y_pred_clipped).map(|p| p.ln());
        
        -(pos_loss + neg_loss).mean().unwrap()
    }

    // Prediction probabilities
    fn predict_proba(&self, X: &Array2<f64>) -> Array1<f64> {
        let (_, _, a2) = self.forward(X);
        a2.column(0).to_owned()
    }

    // Binary predictions
    fn predict(&self, X: &Array2<f64>, threshold: f64) -> Array1<u8> {
        self.predict_proba(X)
            .map(|&p| if p >= threshold { 1 } else { 0 })
    }
}

// Accuracy metric
fn accuracy(y_true: &Array1<u8>, y_pred: &Array1<u8>) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| a == b)
        .count();
    correct as f64 / y_true.len() as f64
}

// Precision metric
fn precision(y_true: &Array1<u8>, y_pred: &Array1<u8>) -> f64 {
    let tp = y_true.iter().zip(y_pred.iter())
        .filter(|(&t, &p)| t == 1 && p == 1).count() as f64;
    let fp = y_true.iter().zip(y_pred.iter())
        .filter(|(&t, &p)| t == 0 && p == 1).count() as f64;
    
    if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) }
}

// Recall metric
fn recall(y_true: &Array1<u8>, y_pred: &Array1<u8>) -> f64 {
    let tp = y_true.iter().zip(y_pred.iter())
        .filter(|(&t, &p)| t == 1 && p == 1).count() as f64;
    let fn_count = y_true.iter().zip(y_pred.iter())
        .filter(|(&t, &p)| t == 1 && p == 0).count() as f64;
    
    if tp + fn_count == 0.0 { 0.0 } else { tp / (tp + fn_count) }
}

// F1 Score
fn f1_score(precision: f64, recall: f64) -> f64 {
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * (precision * recall) / (precision + recall)
    }
}

// CSV row structure
#[derive(Debug, Deserialize)]
struct Row {
    age: f64,
    hypertension: f64,
    heart_disease: f64,
    avg_glucose_level: f64,
    bmi: f64,
    stroke: u8,
}

// CSV loader
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
    println!("=== Neural Network for Stroke Prediction ===\n");
    
    // 1) Load data
    let (X, y, y_label) = load_csv("./BMI_Stroke.csv")?;
    println!("Loaded dataset with {} samples and {} features", X.nrows(), X.ncols());

    // 2) Standardization (Z-score normalization)
    let mean = X.mean_axis(Axis(0)).unwrap();
    let std = X.std_axis(Axis(0), 0.0);
    let X_std = X
        .rows()
        .into_iter()
        .map(|row| &row.to_owned() - &mean)
        .map(|row| &row / &(&std + 1e-9))
        .collect::<Vec<_>>();
    let X_std = Array2::from_shape_vec((X.nrows(), X.ncols()), X_std.into_iter().flatten().collect()).unwrap();

    // 3) Train/test split (80/20)
    let n_train = (X_std.nrows() as f64 * 0.8) as usize;
    let X_train = X_std.slice(s![0..n_train, ..]).to_owned();
    let y_train = y.slice(s![0..n_train]).to_owned();
    let y_train_u8 = y_label.slice(s![0..n_train]).to_owned();

    let X_test = X_std.slice(s![n_train.., ..]).to_owned();
    let y_test_u8 = y_label.slice(s![n_train..]).to_owned();

    println!("Train set: {} samples", X_train.nrows());
    println!("Test set:  {} samples\n", X_test.nrows());

    // 4) Create and train neural network
    let mut model = NeuralNetwork::new(
        5,      // input features
        10,     // hidden neurons
        0.01,   // learning rate
        1000    // epochs
    );
    
    println!("Training Neural Network...");
    println!("Architecture: 5 -> 10 (ReLU) -> 1 (Sigmoid)\n");
    
    model.fit(&X_train, &y_train);

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

    Ok(())
}