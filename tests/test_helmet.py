import pytest

from core.evaluation import AbstainedPrediction, prediction, summarize
from core.helmet import parse_classification, parse_observation


def test_helmet_only_maps_positive_classification_to_ppe():
    result = parse_classification('YES\n')
    assert prediction(result) == (True, ['ppe_violation'])
    assert 'overall_safety_score' not in result
    assert result['requires_human_review']
    assert prediction(parse_classification('No.')) == (False, [])


def test_unknown_is_not_a_negative_prediction():
    result = parse_classification('UNKNOWN')
    with pytest.raises(AbstainedPrediction):
        prediction(result)
    summary = summarize([{'is_hazardous': False, 'error': 'unknown', 'abstained': True}])
    assert summary['confusion']['tn'] == 0
    assert summary['abstentions'] == 1 and summary['schema_error_count'] == 0


@pytest.mark.parametrize('answer', ['', 'YES, but no helmets are missing', 'NO dangerous machines', '{"decision":"clear"}'])
def test_unrecognized_or_contradictory_text_fails_closed(answer):
    with pytest.raises(ValueError):
        parse_classification(answer)


def test_truncated_json_and_additional_fields_are_not_repaired():
    for text in ['{"decision":"clear","evidence":"truncated',
                 '{"decision":"clear","evidence":"helmets","score":100}']:
        with pytest.raises(ValueError):
            parse_observation(text)
