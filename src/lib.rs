use language_model::{LMState, LanguageModel};
use std::collections::HashMap;
use std::{cmp::Ordering, collections::VecDeque};

pub struct InputDecoder {
    last_words: VecDeque<String>,
    predictions: Option<Vec<(String, f64)>>,
    max_no_predictions: usize,
    language_model: LanguageModel,
    lm_state: LMState,
}

impl InputDecoder {
    pub fn new(fname_lm: &str) -> Self {
        let last_words = VecDeque::with_capacity(3);
        let predictions = None;
        let max_no_predictions = 3000;
        let language_model = LanguageModel::read(fname_lm).unwrap();
        let lm_state = LMState::default();

        InputDecoder {
            last_words,
            predictions,
            max_no_predictions,
            language_model,
            lm_state,
        }
    }

    pub fn reset(&mut self) {
        self.lm_state = LMState::default();
        self.last_words.clear();
        self.predictions = None;
    }

    pub fn entered_word(&mut self, word: &str) {
        let word = word.to_ascii_lowercase();
        if self.last_words.len() == 3 {
            self.last_words.pop_front();
        }
        self.lm_state = self.language_model.get_next_state(self.lm_state, &word);
        self.last_words.push_back(word);
        self.predictions = None;
    }

    pub fn get_predictions(&mut self) -> Vec<(String, f64)> {
        if let Some(predictions) = &self.predictions {
            predictions.clone()
        } else {
            self.update_predictions()
        }
    }

    fn update_predictions(&mut self) -> Vec<(String, f64)> {
        let predictions_refs = self
            .language_model
            .predict(self.lm_state, self.max_no_predictions);
        let mut predictions_owned = Vec::new();
        for (word, prob) in predictions_refs {
            predictions_owned.push((word.to_string(), prob as f64));
        }
        self.predictions = Some(predictions_owned.clone());
        predictions_owned
    }

    pub fn find_similar_words(&mut self, query_path: &Vec<(f64, f64)>) -> Vec<(String, f64)> {
        let mut dtw_dist;
        let mut k_best: Vec<(String, f64)> =
            vec![(String::new(), f64::INFINITY); self.max_no_predictions]; // Stores the k nearest neighbors (location, DTW distance)
        let mut bsf = k_best[self.max_no_predictions - 1].1;

        let mut candidate_path;

        let predictions = self.get_predictions();

        // Compare the paths of each word
        for (candidate_word, _) in &predictions {
            candidate_path = path_gen::get_path(&candidate_word);

            dtw_dist = dtw::ucr_improved::dtw(
                &candidate_path,
                &query_path,
                None,
                usize::MAX - 1,
                bsf,
                &dist_points,
            );
            if dtw_dist < bsf {
                let candidate: String = candidate_word.to_owned();
                knn_dtw::ucr::insert_into_k_bsf((candidate, dtw_dist), &mut k_best);
                bsf = k_best[self.max_no_predictions - 1].1;
            }
        }

        /*let mut final_probabilities = HashMap::new();
        for (word, log_prob) in predictions {
            final_probabilities.insert(word, log_prob);
        }
        for (word, dtw_dist) in k_best {
            // The probability is calculated by from the DTW distance
            *final_probabilities.entry(word).or_insert(f64::MIN) += ((bsf - dtw_dist) / bsf).ln();
        }
        let mut final_probabilities: Vec<(String, f64)> = final_probabilities.drain().collect();
        final_probabilities.sort_by(|(_, prob_a), (_, prob_b)| {
            prob_b.partial_cmp(prob_a).unwrap_or(Ordering::Equal)
        });
        final_probabilities*/
        println!("k_best gestures:");
        for (word, _) in &k_best {
            println!("{}", word);
        }

        k_best
    }
}

fn dist_points(a: &(f64, f64), b: &(f64, f64)) -> f64 {
    f64::sqrt((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2))
}
