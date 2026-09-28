import math
import numpy as np
import pytest
from scripts.recommendation_evaluation import ranking_metrics, vector_health, structural_scenarios, parse_junit, model_split_audit


def test_metrics_have_hand_calculated_oracle():
    r = ranking_metrics(['b','a'], {'a':3,'b':0,'c':2}, k=2)
    assert r['precision_at_k'] == .5
    assert r['recall_at_k'] == .5
    assert r['ndcg_at_k'] == pytest.approx((7/math.log2(3))/(7+3/math.log2(3)))

@pytest.mark.parametrize('ranked,labels', [([],{}),(['x'],{})])
def test_no_evidence_never_becomes_perfect_score(ranked,labels):
    assert ranking_metrics(ranked,labels)['status'] == 'not_evaluable'

def test_short_page_is_not_precision_one():
    assert ranking_metrics(['a'], {'a':3}, k=10)['precision_at_k'] == .1

@pytest.mark.parametrize('ranked,labels,k', [(['a','a'],{'a':3},10),(['a'],{'a':4},10),(['a'],{'a':3},0)])
def test_invalid_evaluations_rejected(ranked,labels,k):
    with pytest.raises(ValueError): ranking_metrics(ranked,labels,k)

@pytest.mark.parametrize('a', [np.zeros((2,1024)),np.ones((2,8)),np.full((2,1024),np.nan),np.empty((0,1024))])
def test_vector_health_detects_corruption(a):
    assert vector_health(a)['status']=='invalid'

def test_vector_health_flags_collapse_and_scale():
    a = np.ones((3,1024))
    r = vector_health(a)
    assert r['near_duplicate_pairs']==3 and r['norm_max']==32

def test_sixty_structural_stress_runs():
    cases = structural_scenarios()
    assert sum(c['runs'] for c in cases)==60
    assert all(c['status']=='pass' for c in cases), cases

def test_junit_preserves_failed_and_skipped(tmp_path):
    p=tmp_path/'r.xml'
    p.write_text('<testsuite><testcase name="ok"/><testcase name="bad"><failure/></testcase><testcase name="skip"><skipped/></testcase></testsuite>')
    assert parse_junit(p)['counts']=={'passed':1,'failed':1,'skipped':1}

def test_model_audit_is_not_a_quality_score(tmp_path):
    p=tmp_path/'m.txt';p.write_text('Tree=0\nsplit_feature=0 20 20 30\n')
    assert model_split_audit(p)['personalization_splits_20_30']==3


def test_ann_exact_uses_identical_query_and_disables_quantization():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from qdrant_client.models import SearchParams
    from scripts.recommendation_evaluation import compare_ann_exact
    c=Mock(); c.query_points.side_effect=[SimpleNamespace(points=[SimpleNamespace(id=x) for x in [1,3]]),SimpleNamespace(points=[SimpleNamespace(id=x) for x in [1,2]])]
    r=compare_ann_exact(c,'papers',[.1,.2],SearchParams(),k=2)
    assert r['index_recall_at_k']==.5
    calls=c.query_points.call_args_list
    assert calls[0].kwargs['query']==calls[1].kwargs['query']
    assert calls[1].kwargs['search_params'].exact
    assert calls[1].kwargs['search_params'].quantization.ignore


def test_human_judgments_reject_time_leakage(tmp_path):
    import json
    from scripts.recommendation_evaluation import evaluate_judgment_file
    p=tmp_path/'j.json'
    case={'case_id':'profile1','system':'baseline','ranked':['a'], 'judgments':{'a':3},
          'as_of':'2026-09-25T00:00:00+00:00','latest_source_observed_at':'2026-09-26T00:00:00+00:00'}
    p.write_text(json.dumps([case]))
    with pytest.raises(ValueError):evaluate_judgment_file(p)
    case['latest_source_observed_at']='2026-09-24T00:00:00+00:00'
    p.write_text(json.dumps([case]))
    assert evaluate_judgment_file(p)[0]['precision_at_k']==.1


def test_fully_judged_bad_page_has_zero_precision_not_missing_result():
    r=ranking_metrics(['a'],{'a':0})
    assert r['status']=='measured' and r['precision_at_k']==0
    assert r['ndcg_at_k'] is None and r['recall_at_k'] is None
