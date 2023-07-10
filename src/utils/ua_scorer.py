from src.utils.coval.ua import reader
from src.utils.coval.eval import evaluator
from src.utils.coval.eval.evaluator import evaluate_non_referrings

# based on: https://github.com/juntaoy/universal-anaphora-scorer/blob/main/ua-scorer.py
def eval_coref_ua_scorer(coref_gold, coref_sys):
    metric_dict = {
        "lea": evaluator.lea, "muc": evaluator.muc,
        "bcub": evaluator.b_cubed, "ceafe": evaluator.ceafe,
        "ceafm":evaluator.ceafm, "blanc":[evaluator.blancc,evaluator.blancn]}
    metrics = [(k, metric_dict[k]) for k in metric_dict]
    # TA: pass these as parameters (too many?)
    # or create a new config for coreference evaluation
    keep_singletons = True
    keep_split_antecedent = True
    use_MIN = False
    keep_non_referring = False
    keep_bridging = False
    only_split_antecedent = False
    evaluate_discourse_deixis = False

    metrics_out = evaluate(coref_gold, coref_sys, metrics, keep_singletons, keep_split_antecedent, keep_bridging, keep_non_referring, only_split_antecedent, evaluate_discourse_deixis,  use_MIN)
    return metrics_out

def evaluate(key_file, sys_file, metrics, keep_singletons, keep_split_antecedent, keep_bridging,
    keep_non_referring, only_split_antecedent,evaluate_discourse_deixis, use_MIN):

    doc_coref_infos, doc_non_referring_infos, doc_bridging_infos = reader.get_coref_infos(key_file, sys_file, keep_singletons,
      keep_split_antecedent, keep_bridging, keep_non_referring,evaluate_discourse_deixis,use_MIN, print_debug=True)

    metrics_dict = {}
    for name, metric in metrics:
        recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos, metric, beta=1, only_split_antecedent=only_split_antecedent)
        metrics_dict[name] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}
    return metrics_dict
