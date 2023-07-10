# based on: https://github.com/juntaoy/universal-anaphora-scorer/blob/main/ua-scorer.py
from src.utils.coval.ua import reader
from src.utils.coval.eval import evaluator
from src.utils.coval.eval.evaluator import evaluate_non_referrings


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

    msg = "coreferent markables"
    if keep_singletons:
        msg+= ", singletons"
    if keep_split_antecedent:
        msg+=", split-antecedents"
    if keep_non_referring:
        msg+=", non-referring mentions"
    if keep_bridging:
        msg+=", bridging relations"

    metrics_out = evaluate(coref_gold, coref_sys, metrics, keep_singletons, keep_split_antecedent, keep_bridging, keep_non_referring, only_split_antecedent, evaluate_discourse_deixis,  use_MIN)
    return metrics_out

def evaluate(key_file, sys_file, metrics, keep_singletons, keep_split_antecedent, keep_bridging,
    keep_non_referring, only_split_antecedent,evaluate_discourse_deixis, use_MIN):

    doc_coref_infos, doc_non_referring_infos, doc_bridging_infos = reader.get_coref_infos(key_file, sys_file, keep_singletons,
      keep_split_antecedent, keep_bridging, keep_non_referring,evaluate_discourse_deixis,use_MIN, print_debug=True)

    conll = 0
    conll_subparts_num = 0
    metrics_dict = {}

    for name, metric in metrics:
        recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos, metric, beta=1, only_split_antecedent=only_split_antecedent)
        if name in ["muc", "bcub", "ceafe"]:
            conll += f1
            conll_subparts_num += 1

        print(name)
        print("Recall: %.2f" % (recall * 100), " Precision: %.2f" % (precision * 100), " F1: %.2f" % (f1 * 100))

        metrics_dict[name] = {"precision": precision, "recall": recall, "f1": f1}

    if conll_subparts_num == 3:
        conll = (conll / 3) * 100
        print("CoNLL score: %.2f" % conll)

    if keep_non_referring:
        recall, precision, f1 = evaluate_non_referrings(doc_non_referring_infos)
        print("============================================")
        print("Non-referring markable identification scores:")
        print("Recall: %.2f" % (recall * 100), " Precision: %.2f" % (precision * 100),
" F1: %.2f" % (f1 * 100))
    if keep_bridging:
        score_ar, score_fbm, score_fbe = evaluator.evaluate_bridgings(doc_bridging_infos)
        recall_ar, precision_ar, f1_ar = score_ar
        recall_fbm, precision_fbm, f1_fbm = score_fbm
        recall_fbe, precision_fbe, f1_fbe = score_fbe

        print("============================================")
        print("Bridging anaphora recognition scores:")
        print("Recall: %.2f" % (recall_ar * 100), " Precision: %.2f" % (precision_ar * 100), " F1: %.2f" % (f1_ar * 100))
        print("Full bridging scores (Markable Level):")
        print("Recall: %.2f" % (recall_fbm * 100), " Precision: %.2f" % (precision_fbm * 100), " F1: %.2f" % (f1_fbm * 100))
        print("Full bridging scores (Entity Level):")
        print("Recall: %.2f" % (recall_fbe * 100), " Precision: %.2f" % (precision_fbe * 100), " F1: %.2f" % (f1_fbe * 100))
    return metrics_dict
