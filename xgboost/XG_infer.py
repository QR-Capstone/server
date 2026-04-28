"""Inference CLI for the URL XGBoost classifier."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from XG_core import (
    _validate_single_input_url,
    build_all_explanations,
    load_bundle,
    predict_url,
    predict_url_dom,
)

def predict_url_domain(bundle, url: str):
    return predict_url(
        bundle,
        url,
        enable_domain_age=True,
        enable_ssl=bool(bundle.meta.get("enable_ssl", False)),
        domain_only=True,
    )


def _cmd_predict_url(args: argparse.Namespace) -> int:
    if not 0.0 <= args.threshold <= 1.0:
        print("[Input Error]")
        print("threshold must be between 0.0 and 1.0")
        return 2

    try:
        url = _validate_single_input_url(args.url)
    except ValueError as e:
        print("[Input URL Error]")
        print(str(e))
        return 2

    bundle_typo = load_bundle(args.model_typo)
    bundle_domain = load_bundle(args.model_domain)
    bundle_dom = load_bundle(args.model_dom)
    _, prob_typo, typo_feature_map = predict_url(
        bundle_typo,
        url,
        enable_domain_age=False,
        enable_ssl=False,
        domain_only=False,
    )
    _, prob_domain, domain_feature_map = predict_url_domain(bundle_domain, url)
    _, prob_dom, dom_feature_map = predict_url_dom(bundle_dom, url)
    final_probability = max(prob_typo, prob_domain, prob_dom)
    if prob_typo >= 0.90 or prob_domain >= 0.90:
        verdict_label = 1
    elif (
        (prob_typo >= 0.5 and prob_domain >= 0.5)
        or (prob_typo >= 0.5 and prob_dom >= 0.5)
        or (prob_domain >= 0.5 and prob_dom >= 0.5)
    ):
        verdict_label = 1
    else:
        verdict_label = 0
    verdict = "malicious" if verdict_label == 1 else "benign"
    explanations = build_all_explanations(
        url=url,
        typo_feat_map=typo_feature_map,
        typo_probability=prob_typo,
        domain_feat_map=domain_feature_map,
        domain_probability=prob_domain,
        dom_feature_map=dom_feature_map,
        dom_probability=prob_dom,
    )

    print("[Input URL]")
    print(url)
    print()
    print("[Detailed Reasons / 상세 근거]")
    for reason in explanations:
        print(f"- {reason}")
    print()
    print("[Model Outputs]")
    print(f"typo_probability: {prob_typo:.4f}")
    print(f"domain_probability: {prob_domain:.4f}")
    print(f"dom_probability: {prob_dom:.4f}")
    print(f"final_probability: {final_probability:.4f}")
    print()
    print("[DOM Features]")
    print(f"dom_max_depth: {dom_feature_map['dom_max_depth']:.4f}")
    print(f"dead_link_ratio: {dom_feature_map['dead_link_ratio']:.4f}")
    print(f"hidden_tags_count: {dom_feature_map['hidden_tags_count']:.4f}")
    print(f"suspicious_form_action: {dom_feature_map['suspicious_form_action']:.4f}")
    print(f"dom_fetch_failed: {dom_feature_map['dom_fetch_failed']:.4f}")
    print()
    print("[Prediction Result]")
    print(f"threshold: {args.threshold:.2f}")
    print(f"verdict: {verdict}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xgboost_infer.py",
        description="Run inference for a single URL with typo and domain-age XGBoost bundles.",
    )
    p.add_argument(
        "--model_typo",
        default="url_xgb_paired_first.joblib",
        help="Path to typo model bundle (.joblib). Default: url_xgb_paired_first.joblib",
    )
    p.add_argument(
        "--model_domain",
        default="url_xgb_domain_age.joblib",
        help="Path to domain-age model bundle (.joblib). Default: url_xgb_domain_age.joblib",
    )
    p.add_argument(
        "--model_dom",
        default="url_xgb_dom.joblib",
        help="Path to DOM model bundle (.joblib). Default: url_xgb_dom.joblib",
    )
    p.add_argument("--url", required=True, help="Single URL to classify.")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classifying as malicious.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_argparser()
    args = parser.parse_args(argv)
    return _cmd_predict_url(args)


if __name__ == "__main__":
    raise SystemExit(main())
