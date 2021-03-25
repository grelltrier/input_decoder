use super::*;

#[test]
fn test_load_model() {
    let fname_lm = "language_model.bin";
    let key_layout = path_gen::get_default_buttons_centers();
    let max_no_predictions = 2;
    InputDecoder::new(fname_lm, key_layout, max_no_predictions);
}

#[test]
fn test_get_predictions() {
    let fname_lm = "language_model.bin";
    let key_layout = path_gen::get_default_buttons_centers();
    let max_no_predictions = 2;
    let mut input_decoder = InputDecoder::new(fname_lm, key_layout, max_no_predictions);

    // Go to state 5
    input_decoder.entered_word("b");
    input_decoder.entered_word("b");
    let predictions = input_decoder.get_predictions();
    let correct_predictions = vec!["a", "b"];

    assert!(predictions == correct_predictions);

    // Reset and go to state 1
    input_decoder.reset();
    input_decoder.entered_word("a");
    let predictions = input_decoder.get_predictions();
    let correct_predictions = vec!["b", "a"];
    assert!(predictions == correct_predictions);
}

#[test]
fn test_get_similar_word() {
    let fname_lm = "language_model.bin";
    let key_layout = path_gen::get_default_buttons_centers();
    let max_no_predictions = 2;
    let input_decoder = InputDecoder::new(fname_lm, key_layout, max_no_predictions);

    // Get similar word (input is exactly the center of the "a" key)
    let query_path = vec![(0.100, 0.15)];
    let similar_word = input_decoder.find_similar_words(&query_path);
    assert!(similar_word[0].0 == "a".to_string());

    // Get similar word (input is exactly the center of the "b" key)
    let query_path = vec![(0.550, 0.25)];
    let similar_word = input_decoder.find_similar_words(&query_path);
    assert!(similar_word[0].0 == "b".to_string());

    // Get similar word (input is close to the center of the "b" key)
    let query_path = vec![(0.500, 0.252)];
    let similar_word = input_decoder.find_similar_words(&query_path);
    assert!(similar_word[0].0 == "b".to_string());
}
