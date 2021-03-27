use language_model::{LMState, LanguageModel};
use path_gen::WordPath;
use std::collections::HashMap;

#[cfg(test)]
mod tests;

pub struct InputDecoder {
    max_no_predictions: usize,
    language_model: LanguageModel,
    lm_state: LMState,
    key_layout: HashMap<String, (f64, f64)>,
}

impl InputDecoder {
    /// Create a new InputDecoder struct
    /// A file name to the language model and the maximum
    /// number of predictions must be provided
    pub fn new(
        fname_lm: &str,
        key_layout: HashMap<String, (f64, f64)>,
        max_no_predictions: usize,
    ) -> Self {
        let language_model = LanguageModel::read(fname_lm).unwrap();
        let lm_state = LMState::default();

        InputDecoder {
            max_no_predictions,
            language_model,
            lm_state,
            key_layout,
        }
    }

    /// Reset the InputDecoder to its initial state
    pub fn reset(&mut self) {
        self.lm_state = LMState::default();
    }

    /// Take a word as input
    pub fn entered_word(&mut self, word: &str) {
        let word = word.to_ascii_lowercase();
        self.lm_state = self.language_model.get_next_state(self.lm_state, &word);
    }

    /// Get a prediction for the next word
    pub fn get_predictions(&self) -> Vec<String> {
        let predictions: Vec<String> = self
            .language_model
            .predict(self.lm_state, self.max_no_predictions)
            .iter()
            .map(|(word, _)| word.to_string())
            .collect();
        predictions
    }

    /// Find the most similar word for the provided path
    /// Only the most likely next words are considered
    /// The method uses DTW to calculate the similarity
    pub fn find_similar_words(&self, query_path: &Vec<(f64, f64)>) -> Vec<(String, f64)> {
        let mut dtw_dist;
        let k = 10;
        let mut k_best: Vec<(String, f64)> = vec![(String::new(), f64::INFINITY); k]; // Stores the k nearest neighbors (location, DTW distance)
        let mut bsf = k_best[k - 1].1;

        let w = (query_path.len() as f64 * 0.1).round() as usize;

        // In order to better compare the drawn path with the ideal path, we want them to have a similar density of points
        // To generate ideal paths with a similar density, we calculate the density of the drawn path
        let desired_point_density = {
            let mut drawn_path_length = 0.0;
            let mut drawn_path_iter = query_path.iter().peekable();
            let mut leg_dist;
            // Calculate the length of the drawn path
            while let Some(start_point) = drawn_path_iter.next() {
                if let Some(&end_point) = drawn_path_iter.peek() {
                    leg_dist = dist_points(start_point, end_point);
                    drawn_path_length += leg_dist;
                }
            }
            drawn_path_length / query_path.len() as f64
        };

        let predictions = self
            .language_model
            .predict(self.lm_state, self.max_no_predictions);
        let mut candidate_path;
        let mut word_path;
        // Compare the paths of each word
        for &(candidate_word, _) in &predictions {
            word_path = WordPath::new(&self.key_layout, candidate_word);

            let (candidate_first, candidate_last) = word_path.get_first_last_points();

            // Use lower bound of Kim to skip impossible candidates
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

            // The candidate counld not be skipped so the full path is generated
            candidate_path = if let Some(candidate_path) = word_path.get_path(desired_point_density)
            {
                candidate_path
            } else {
                continue;
            };

            // Calculate the similarity
            dtw_dist = dtw::rpruned::dtw(&candidate_path, &query_path, None, w, bsf, &dist_points);

            // If the candidate is a better match, save it
            if dtw_dist < bsf {
                let candidate: String = candidate_word.to_owned();
                knn_dtw::insert_into_k_bsf((candidate, dtw_dist), &mut k_best);
                bsf = k_best[k - 1].1;
            }
        }
        k_best
    }
}

fn dist_points(start: &(f64, f64), end: &(f64, f64)) -> f64 {
    f64::sqrt((start.0 - end.0).powi(2) + (start.1 - end.1).powi(2))
}
