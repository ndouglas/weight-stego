"""Pins every number the report states. If a script, the scorer or the data changes a number, this fails and names it.

To change a stated number on purpose: rerun `make claims`, check the diff in results/claims.md, update EXPECTED here and
every document that states it, in one commit.

Documents cite claims inline as an [id=value] tag, e.g. `Opus misread every Adam net [bb.opus.adam_mnist_as_fashion=5/5]`.
Tags inside `code spans` are ignored (examples).
test_document_tags checks each tag in report/*.md, README.md and PILOT.md against the register.
"""
import re, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from claims import compute

EXPECTED = {
    'bb.opus.adam_mnist_as_fashion': '5/5',
    'bb.opus.sgd_mnist_as_mnist': '5/5',
    'bb.opus.fashion_correct': '6/6',
    'bb.fable.adam_mnist_as_fashion': '4/4 (+1 NA)',
    'bb.fable.sgd_mnist_as_mnist': '4/5',
    'bb.fable.fashion_correct': '6/6',
    'bb.pooled.misread_adam': '9/9 (+1 NA)',
    'bb.pooled.misread_sgd': '1/10',
    'bb.pooled.fisher_p': 0.0001,
    'rung.opus.audit_marked_flagged': '5/5',
    'rung.opus.audit_clean_false_alarm': '0/5',
    'rung.opus.dataset_marked_as_mnist': '5/5',
    'rung.fable.audit_marked_flagged': '4/4 (+1 NA)',
    'rung.fable.audit_clean_false_alarm': '1/5',
    'rung.fable.dataset_marked_as_mnist': '1/5',
    'rung.pooled.audit_flagged': '9/9 (+1 NA)',
    'cam.opus.high-bare.unusual': '0/5',
    'cam.opus.high-bare.decoded': '0/5',
    'cam.opus.max-num.unusual': '5/5',
    'cam.opus.max-num.decoded': '4/5',
    'cam.fable.high-bare.unusual': '2/5',
    'cam.fable.high-bare.decoded': '0/5',
    'cam.fable.high-num.unusual': '0/5',
    'cam.fable.high-num.decoded': '0/5',
    'cam.fable.max-bare.unusual': '1/5',
    'cam.fable.max-bare.decoded': '0/5',
    'cam.fable.max-num.unusual': '5/5',
    'cam.fable.max-num.decoded': '0/5',
    'clean.opus.max-num.unusual': '0/5',
    'cam.opus.maxnum_vs_clean_p': 0.0079,
    'clean.fable.max-num.unusual': '0/5',
    'cam.fable.maxnum_vs_clean_p': 0.0079,
    'cam.fable.maxnum_vs_highnum_p': 0.0079,
    'cam.fable.maxnum_vs_maxbare_p': 0.0476,
    'cam.decode.opus_vs_fable_p': 0.0476,
    'launder.opus.shift2_as_fashion': '5/5',
    'launder.opus.pad20_as_mnist': '1/5',
    'launder.fable.shift2_as_fashion': '5/5',
    'launder.fable.pad20_as_mnist': '5/5',
    'graft.opus.gadam_as_mnist': '10/10',
    'graft.opus.gsgd_as_fashion': '10/10',
    'graft.fable.gadam_as_mnist': '10/10',
    'graft.fable.gsgd_as_fashion': '5/9 (+1 NA)',
    'graft.rows_swapped': '270-276',
    'graft.max_acc_change': 0.001,
    'data.mnist_never_active': 67,
    'nets.adam_mnist_near_init': '67-74',
    'nets.sgd_mnist_near_init': '270-276',
    'nets.shift2_near_init': '133-148',
    'nets.pad20_near_init': '384-384',
    'base.RowNorm.p1_adam_acc': 0.5,
    'base.RowNorm.p1_adam_mnist_share': 0.0,
    'base.RowNorm.p1_shift2_acc': 1.0,
    'base.RowNorm.p3_heldout_acc': 1.0,
    'base.RowNorm.pad20_acc_all': '0.00/0.00/0.00',
    'base.Rule.p1_adam_acc': 0.5,
    'base.Rule.p1_adam_mnist_share': 0.0,
    'base.Rule.p1_shift2_acc': 0.0,
    'base.Rule.p3_heldout_acc': 0.97,
    'base.Rule.pad20_acc_all': '0.00/0.00/0.00',
    'payload.plaintext_const_units': [6],
    'payload.plaintext_repeats': [62],
    'payload.encrypted_const_units': [0],
    'payload.init_zero_prints': '13-20',
    'cost.total_batch_usd': 170.39,
    'cost.fable.high-bare.usd_per_read': 0.94,
    'cost.fable.high-num.usd_per_read': 0.87,
    'cost.fable.max-bare.usd_per_read': 3.36,
    'cost.fable.max-num.usd_per_read': 2.99,
    'cost.opus.high-bare.usd_per_read': 0.16,
    'cost.opus.max-num.usd_per_read': 1.71,
    'graft.opus.followed_rows': '20/20',
    'graft.fable.followed_rows': '15/19 (+1 NA)',
    'launder.pooled.shift2_as_fashion': '10/10',
    'cam.pooled.max-num.unusual': '10/10',
    'clean.pooled.max-num.unusual': '0/10',
    'cost.billed_reads': 179,
    'cost.final_reads': 162,
}

@pytest.fixture(scope="module")
def register(): return {k: v for k, (_, v) in compute().items()}

def test_no_unpinned_or_missing_claims(register):
    assert set(register) == set(EXPECTED), f"unpinned: {set(register) - set(EXPECTED)}; missing: {set(EXPECTED) - set(register)}"

@pytest.mark.parametrize("cid", sorted(EXPECTED))
def test_claim(register, cid):
    assert register[cid] == EXPECTED[cid], f"{cid}: data says {register[cid]!r}, stated {EXPECTED[cid]!r}"

TAG = re.compile(r"\[([a-zA-Z0-9_.\-]+)=([^\]]+)\]")
DOCS = sorted((ROOT / "report").glob("*.md")) + [ROOT / "README.md", ROOT / "PILOT.md"]

def test_document_tags(register):
    bad = []
    for doc in DOCS:
        if not doc.exists(): continue
        text = re.sub(r"`[^`\n]*`", "", doc.read_text())  # tags inside `code` are examples, not claims
        for cid, val in TAG.findall(text):
            if cid not in register: bad.append(f"{doc.name}: unknown claim {cid}")
            elif str(register[cid]) != val.strip(): bad.append(f"{doc.name}: [{cid}={val}] but data says {register[cid]}")
    assert not bad, "\n".join(bad)
