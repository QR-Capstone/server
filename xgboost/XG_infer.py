"""Inference CLI for the URL XGBoost classifier."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from XG_core import (
    _validate_single_input_url,
    build_all_explanations,
    load_bundle,
    predict_url,
    predict_url_dom,
)

_TYPO_FEATURE_INCLUDE_PREFIXES: Sequence[str] = ("brand_", "sld_")
_TYPO_FEATURE_INCLUDE_KEYS: Set[str] = {
    "host_contains_brand_token",
    "has_hyphen",
    "has_ip_host",
    "has_multi_level_tld",
    "has_country_code_tld",
    "has_safe_second_level_hint",
    "has_redirect_pattern",
    "has_do_or_html_endpoint",
    "has_api_keyword",
    "has_cdn_keyword",
    "host_len",
    "host_length",
    "is_short_domain",
    "is_public_hosting_platform",
    "num_at",
    "num_hyphens",
    "num_subdomains",
    "num_underscores",
    "path_depth",
    "domain_digit_ratio",
    "domain_homoglyph_ratio",
    "domain_letter_digit_alternations",
    "domain_vowel_like_digit_count",
}
_TYPO_FEATURE_EXCLUDE_PREFIXES: Sequence[str] = (
    "domain_age_",
    "rdap_status_",
    "domain_is_",
    "ssl_",
)
_TYPO_FEATURE_EXCLUDE_KEYS: Set[str] = {"rdap_creation_date_iso"}

_DOMAIN_FEATURE_INCLUDE_KEYS: Set[str] = {
    "domain_age_days",
    "domain_age_log_days",
    "domain_age_missing",
    "domain_is_very_new",
    "domain_is_new",
    "domain_is_established",
    "domain_is_old",
    "rdap_creation_date_iso",
    "rdap_status_ok",
    "rdap_status_lookup_failed",
    "rdap_status_not_registered",
    "rdap_status_parse_failed",
    "ssl_valid_days",
    "ssl_remaining_days",
    "ssl_age_days",
    "ssl_missing",
    "ssl_status_lookup_failed",
    "ssl_status_no_cert",
    "ssl_is_short_lived",
    "ssl_is_normal_lived",
    "ssl_is_long_lived",
    "ssl_is_very_new",
    "ssl_is_recent",
    "ssl_is_mature",
    "ssl_expires_very_soon",
    "ssl_expires_soon",
    "ssl_expires_far",
}
_DOMAIN_FEATURE_EXCLUDE_PREFIXES: Sequence[str] = (
    "brand_",
    "sld_",
    "has_",
    "num_",
)
_DOMAIN_FEATURE_EXCLUDE_KEYS: Set[str] = {
    "host_contains_brand_token",
    "host_len",
    "host_length",
    "path_depth",
    "is_short_domain",
    "is_public_hosting_platform",
    "domain_digit_ratio",
    "domain_homoglyph_ratio",
    "domain_letter_digit_alternations",
    "domain_vowel_like_digit_count",
}


def _print_sorted_feature_map(feature_map: Dict[str, Any]) -> None:
    for key in sorted(feature_map.keys()):
        value = feature_map[key]
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def _print_feature_subset(
    feature_map: Dict[str, Any],
    *,
    include_prefixes: Optional[Sequence[str]] = None,
    include_keys: Optional[Iterable[str]] = None,
    exclude_prefixes: Optional[Sequence[str]] = None,
    exclude_keys: Optional[Iterable[str]] = None,
) -> None:
    """Print a filtered view of feature_map without mutating it."""
    ipfx = tuple(include_prefixes or ())
    ikeys = frozenset(include_keys or ())
    epfx = tuple(exclude_prefixes or ())
    ekeys = frozenset(exclude_keys or ())
    use_include = bool(ipfx or ikeys)

    def _included(key: str) -> bool:
        if use_include:
            if key in ikeys:
                pass
            elif any(key.startswith(p) for p in ipfx):
                pass
            else:
                return False
        if key in ekeys:
            return False
        if any(key.startswith(p) for p in epfx):
            return False
        return True

    subset = {k: v for k, v in feature_map.items() if _included(k)}
    _print_sorted_feature_map(subset)


def _xgboost_weighted_ensemble_verdict_label(
    prob_typo: float, prob_domain: float, prob_dom: float
) -> tuple[float, int]:
    """Weighted soft score + strong single-channel gates; returns (final_score, 1=malicious)."""
    final_score = prob_typo * 0.50 + prob_domain * 0.35 + prob_dom * 0.15
    if prob_typo >= 0.70 or prob_domain >= 0.80 or prob_dom >= 0.80:
        return final_score, 1
    if final_score >= 0.55:
        return final_score, 1
    return final_score, 0


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
    _, prob_dom, dom_feature_map = predict_url_dom(
        bundle_dom, url, print_dom_feature_debug=False
    )
    final_probability, verdict_label = _xgboost_weighted_ensemble_verdict_label(
        prob_typo, prob_domain, prob_dom
    )
    verdict = "malicious" if verdict_label == 1 else "benign"
    explanations = build_all_explanations(
        url=url,
        typo_feat_map=typo_feature_map,
        typo_probability=prob_typo,
        domain_feat_map=domain_feature_map,
        domain_probability=prob_domain,
        dom_feature_map=dom_feature_map,
        dom_probability=prob_dom,
        verdict_label=verdict_label,
        threshold=args.threshold,
    )

    print()
    print("[Typo Features]")
    print("(URL/lexical feature subset for readability)")
    _print_feature_subset(
        typo_feature_map,
        include_prefixes=_TYPO_FEATURE_INCLUDE_PREFIXES,
        include_keys=_TYPO_FEATURE_INCLUDE_KEYS,
        exclude_prefixes=_TYPO_FEATURE_EXCLUDE_PREFIXES,
        exclude_keys=_TYPO_FEATURE_EXCLUDE_KEYS,
    )
    if args.show_all_features:
        print()
        print("[Typo Features - All Raw]")
        _print_sorted_feature_map(typo_feature_map)
    print()
    print("[Domain Features]")
    print("(RDAP/SSL feature subset used to inspect domain_probability)")
    _print_feature_subset(
        domain_feature_map,
        include_keys=_DOMAIN_FEATURE_INCLUDE_KEYS,
        exclude_prefixes=_DOMAIN_FEATURE_EXCLUDE_PREFIXES,
        exclude_keys=_DOMAIN_FEATURE_EXCLUDE_KEYS,
    )
    if args.show_all_features:
        print()
        print("[Domain Features - All Raw]")
        _print_sorted_feature_map(domain_feature_map)
    print()
    print("[DOM Features]")
    print("(Final DOM feature map used for dom_probability)")
    print(f"dom_max_depth: {dom_feature_map['dom_max_depth']:.4f}")
    print(f"dead_link_ratio: {dom_feature_map['dead_link_ratio']:.4f}")
    print(f"hidden_tags_count: {dom_feature_map['hidden_tags_count']:.4f}")
    print(f"suspicious_form_action: {dom_feature_map['suspicious_form_action']:.4f}")
    print(f"dom_timeout: {dom_feature_map['dom_timeout']:.4f}")
    print(f"dom_ssl_error: {dom_feature_map['dom_ssl_error']:.4f}")
    print(f"dom_blocked: {dom_feature_map['dom_blocked']:.4f}")
    print(f"dom_connection_error: {dom_feature_map['dom_connection_error']:.4f}")
    print(f"dom_connection_reset: {dom_feature_map['dom_connection_reset']:.4f}")
    print(f"dom_dns_failed: {dom_feature_map['dom_dns_failed']:.4f}")
    print(f"dom_connection_refused: {dom_feature_map['dom_connection_refused']:.4f}")
    print()
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
        help="Threshold passed to detailed explanations (ensemble verdict uses weighted score and fixed gates).",
    )
    p.add_argument(
        "--show-all-features",
        action="store_true",
        default=False,
        help="After typo/domain subsets, also print full raw feature maps for debugging.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_argparser()
    args = parser.parse_args(argv)
    return _cmd_predict_url(args)


if __name__ == "__main__":
    raise SystemExit(main())
