"""Phase 2 parity harness (WP-A): offline replay + classified differ.

`replay` rebuilds one season-1 week with the new pipeline entirely offline from
archived season-1 inputs into a scratch output root; `diff` joins the rebuilt
week against the season-1 baseline artifacts and classifies every field delta.
Neither module writes into the real ``out/`` tree.
"""
