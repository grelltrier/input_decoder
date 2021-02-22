use language_model::{LMState, LanguageModel};
use path_gen::WordPath;
use std::collections::HashMap;
use std::collections::VecDeque;

pub struct InputDecoder {
    last_words: VecDeque<String>,
    predictions: Option<Vec<(String, f64)>>,
    max_no_predictions: usize,
    language_model: LanguageModel,
    lm_state: LMState,
    key_layout: HashMap<String, (f64, f64)>,
}

impl InputDecoder {
    pub fn new(fname_lm: &str, max_no_predictions: usize) -> Self {
        let last_words = VecDeque::with_capacity(3);
        let predictions = None;
        let language_model = LanguageModel::read(fname_lm).unwrap();
        let lm_state = LMState::default();
        let key_layout = path_gen::get_default_buttons_centers();

        InputDecoder {
            last_words,
            predictions,
            max_no_predictions,
            language_model,
            lm_state,
            key_layout,
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

    pub fn get_all_words(&self) -> Vec<(String, f64)> {
        let predictions_refs = self.language_model.predict(LMState::default(), usize::MAX);
        let mut predictions_owned = Vec::new();
        for (word, prob) in predictions_refs {
            predictions_owned.push((word.to_string(), prob as f64));
        }
        predictions_owned
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
        let k = 10;
        let mut k_best: Vec<(String, f64)> = vec![(String::new(), f64::INFINITY); k]; // Stores the k nearest neighbors (location, DTW distance)
        let mut bsf = k_best[k - 1].1;

        let w = (query_path.len() as f64 * 0.1).round() as usize;

        println!("drawn path:");
        for (x, y) in query_path {
            println!("{:.3},{:.3}", x, y);
        }

        let predictions = self.get_predictions();
        let mut candidate_path;
        let mut word_path;
        // Compare the paths of each word
        for (candidate_word, _) in &predictions {
            word_path = WordPath::new(&self.key_layout, candidate_word, query_path.len());

            let (candidate_first, candidate_last) = word_path.get_first_last_points();

            if let Some(candidate_first) = candidate_first {
                let candidate_last = if let Some(candidate_last) = candidate_last {
                    candidate_last
                } else {
                    candidate_first
                };

                let mut dist = dist_points(candidate_first, &query_path[0]);
                dist += dist_points(candidate_last, &query_path[query_path.len() - 1]);

                if dist > bsf {
                    continue;
                }
            } else {
                // The candidate word is an empty string
                continue;
            }

            candidate_path = if let Some(candidate_path) = word_path.get_path() {
                candidate_path
            } else {
                continue;
            };

            println!("Candidate word '{}':", candidate_word);

            dtw_dist =
                dtw::ucr_improved::dtw(&candidate_path.0, &query_path, None, w, bsf, &dist_points);
            // We multiply the dtw distance by the distance of the candidates path because shorter paths have a higher sampling rate than longer paths due to the ideal paths always having the same length as the query's path. If a path is twice as long, we multiply the dtw by two, accounting for the higher sampling rate
            dtw_dist *= candidate_path.1;
            if dtw_dist < bsf {
                let candidate: String = candidate_word.to_owned();
                knn_dtw::ucr::insert_into_k_bsf((candidate, dtw_dist), &mut k_best);
                bsf = k_best[k - 1].1;
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
        for (word, _) in k_best.iter().take(10) {
            println!("{}", word);
        }

        k_best
    }
}

fn dist_points(a: &(f64, f64), b: &(f64, f64)) -> f64 {
    f64::sqrt((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2))
}
