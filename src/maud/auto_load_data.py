"""
Semi-automatic parsing of Huggingface Datasets from CSV raw data files.
"""

import csv
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from maud import data, specs

headers = data.get_headers()


class QuestionSpec(NamedTuple):
    """Data structure corresponding to matching columns in the dataset.

    Use questions_from_spec(spec) to generate secondary data structures corresponding to each huggingface
    dataset that can be loaded from these columns.
    """

    context_column: int
    question_columns: Sequence[int]
    second_context_column: Optional[int] = None  # For "Agreement includes.*MAE-Answer"

    @classmethod
    def from_int_seq(cls, question_columns: Sequence[int]) -> "QuestionSpec":
        context_column = question_columns[0] - 1
        assert context_column >= 0
        assert context_column not in question_columns
        return cls(
            context_column=context_column,
            question_columns=question_columns,
        )


QUESTION_SPECS: List[QuestionSpec] = []
"""This list of specs is used to auto-parse Huggingface Datasets from
the CSV files."""


def ADD(spec: QuestionSpec) -> QuestionSpec:
    """Helper function to append something to QUESTION_SPECS.

    Returns the argument for chaining convenience.
    """
    QUESTION_SPECS.append(spec)
    return spec


def grab_matching_range(key1: str, key2: str) -> QuestionSpec:
    """Use regex patterns to find the column indices of the left and
    right side of a range of questions that all use the same context.

    The return value is suitable as an argument to ADD.
    """
    i1 = headers.index(key1)
    i2 = headers.index(key2)
    assert i2 > i1
    rng = range(i1, i2 + 1)
    return QuestionSpec.from_int_seq(rng)


def grab_singleton(key: str) -> QuestionSpec:
    """Use regex patterns to find the column index of a single
    question, assuming that the context column index is one minus the previous
    index.

    The return value is suitable as an argument to ADD.
    """
    i1 = headers.index(key)
    assert i1 is not None, key
    return QuestionSpec.from_int_seq([i1])


def grab_matching_set(keys: List[str]) -> QuestionSpec:
    """Good for question columns that skip indices, e.g. Buyer Full Name.

    The return value is suitable as an argument to ADD.
    """
    out = []
    for k in keys:
        out.append(headers.index(k))
    return QuestionSpec.from_int_seq(out)


def grab_double_context_set(
        pattern: str
) -> QuestionSpec:
    i = headers.index(pattern)
    return QuestionSpec(context_column=i-2, second_context_column=i-1, question_columns=[i])


R = data.regex_match_header


consider_range = ADD(grab_matching_range(
    "Type of Consideration-Answer",
    "Stock Deal: Fixed Ratio v. Fixed Value-Answer",
))
accuracy_range = ADD(grab_matching_range(
    R("Accuracy of Target.*R&W: Bringdown Timing Answer"),
    "Materiality/MAE Scrape applies to",
))
compliance_range = ADD(grab_singleton(
    "Compliance with Target Covenant Closing Condition-Answer",
))
absence_range = ADD(grab_matching_range(
    "Absence of Litigation Closing Condition: Governmental v. Non-Governmental-Answer",
    "Absence of Litigation Closing Condition: Pending v. Threatened v. Threatened in Writing-Answer"))
nomae_range = ADD(grab_double_context_set(R("Agreement includes.*MAE-Answer")))
mae_range = ADD(grab_matching_range(
    R("MAE.*adverse impact.*consummate.*"),
    R("Other carveouts.*modifier"),
))
know_range = ADD(grab_matching_range("Knowledge Definition-Answer", R("Knowledge Definition.*identified.*")))
noshop_range = ADD(grab_matching_range(
    "Liability for breaches of no-shop by Target Representatives (Y/N)",
    "Liability standard for no-shop breach by Target Non-D&O Representatives"))
fid_xcept_range = ADD(grab_matching_range(
    R("Fiduciary exception.*standard.*Answer.*"),
    R("Fiduciary exception.*trigger.*Answer")))
fid_xcept_cor_range = ADD(grab_matching_range(
    R("COR permitted with board fid.*det.*only"),
    R("COR standard.*intervening event.*")))
agree_cor_range = ADD(grab_matching_range(
    "Initial matching rights period (COR)-Answer",
    "Number of additional matching rights periods for modifications (COR)",
))
supoffer_range = ADD(grab_matching_range(
    "Definition includes stock deals-Answer",
    R(".*is the sole consideration")))
interevent_range = ADD(grab_matching_range(
    "Definition contains knowledge requirement - answer",
    "Definition contains a materiality standard (Y/N)"))
ftrtrigger_range = ADD(grab_singleton("FTR Triggers-Answer"))
limitftr_range = ADD(grab_singleton("Limitations on FTR Exercise-Answer"))
agreematch_range = ADD(grab_matching_range(
    "Initial matching rights period (FTR)-Answer",
    "Number of additional matching rights periods for modifications (FTR)"))
acqprop_range = ADD(grab_matching_range("Acquisition Proposal Timing-Answer", "Tail Period Length-Answer"))
genbreach_range = ADD(grab_singleton(
    R(".*Breach required to be willful, material and/or intentional")))
breachshop_range = ADD(grab_singleton("Breach of No Shop required to be willful, material and/or intentional"))
breachmeeting_range = ADD(grab_singleton(
    "Breach of Meeting Covenant required to be willful, material and/or intentional"))
ordinary_range = ADD(grab_matching_range(
    "Buyer consent requirement (ordinary course)-Answer",
    "Ordinary Course Covenant includes carve-out for Pandemic responses-Answer (Y/N)",
))
negative_range = ADD(grab_matching_range(
    "Buyer consent requirement (negative interim covenant)-Answer",
    "Negative Interim Covenant includes carveout for pandemic responses-Answer (Y/N)"
))
genanti_range = ADD(grab_singleton("General Antitrust Efforts Standard-Answer"))
limanti_range = ADD(grab_singleton("Limitations on Antitrust Efforts-Answer"))
specific_range = ADD(grab_singleton("Specific Performance-Answer"))


ANSWER_KEY_BLOCK_LIST = {
    'Buyer Full Name',  # Span extraction task.
    'Target Full Name',  # Span extraction task.
    'Other MAE carveouts',  # Span extraction task.
    'Other carveouts subject to "disproportionate impact" modifier',   # Span extraction task?
    'Limitations on Antitrust Efforts-Answer'
}

MULTI_BINARY_ANSWER_KEYS = {
    R('Accuracy of Fundamental Target.*Types of R.*Ws'),
    "Materiality/MAE Scrape applies to",
    "FLS (MAE) applies to",
    "A/P/C application to-Answer",
    "W/N/A/F applies to-Answer",
    R("W/N/A/F subject to .*disproportionate impact.*-Answer"),
    "Relational language (MAE carveout)-Answer (Dropdown)",
    "Limitations on FTR Exercise-Answer",
    "Acquisition Proposal required to be publicly disclosed-Answer",
    "Acquisition Proposal Timing-Answer",
}


def generate_specs(verbosity: int = 1) -> List[specs.BaseDatasetSpec]:
    auto_qs = []
    auto_rejected_qs = []
    for q in QUESTION_SPECS:
        accepted, rejected = questions_from_spec(q, verbose=verbosity>=2)
        auto_qs.extend(accepted)
        auto_rejected_qs.extend(rejected)

    if verbosity >= 2:
        print(f"Accepted {len(auto_qs)} questions. Rejected {len(auto_rejected_qs)} questions.")

    if verbosity >= 2:
        _debug_write_detailed_logs(auto_qs, auto_rejected_qs)
    return auto_qs


def get_all_valid_specs(verbosity: int = 1) -> List[specs.BaseDatasetSpec]:
    all_specs = generate_specs(verbosity=verbosity)
    valid_specs = []
    for spec in all_specs:
        if not isinstance(spec, specs.MultiBinaryDatasetSpec):
            valid_specs.append(spec)
    return valid_specs


def _debug_write_detailed_logs(
        auto_qs: Sequence[specs.BaseDatasetSpec],
        auto_rejected_qs: Sequence[specs.BaseDatasetSpec],
) -> None:
    with open("scrap/auto_qs.txt", "w") as f:
        for q in auto_qs:
            f.write(f"{repr(q)}\n")
    with open("scrap/auto_answers.txt", "w") as f:
        for q in auto_qs:
            f.write(f"{q.ordered_answers}\n")
    with open("scrap/auto_counter.txt", "w") as f:
        for q in auto_qs:
            counter = q.answer_counter()
            f.write(f"{counter}\n")
    with open("scrap/auto_answer_keys.txt", "w") as f:
        for q in auto_qs:
            f.write(f"{(q.context_key, q.answer_key)}\n")
    with open("scrap/auto_rejected_answer_keys.txt", "w") as f:
        for q in auto_rejected_qs:
            if q.answer_key in ANSWER_KEY_BLOCK_LIST:
                f.write("((MANUAL REJECTION)) ")
            f.write(f"{(q.context_key, q.answer_key)}\n")
    with open("scrap/auto_rejected_answers.txt", "w") as f:
        for q in auto_rejected_qs:
            if q.answer_key in ANSWER_KEY_BLOCK_LIST:
                f.write("((MANUAL REJECTION)) ")
            f.write(f"{q.answer_counter()}\n")


def get_question_reject_reason(
        spec: QuestionSpec,
        q: specs.BaseDatasetSpec,
) -> Optional[str]:
    if isinstance(q, specs.MultiBinaryDatasetSpec):
        # Hack: This isn't actually valid, but we want to appear
        # in the toolings
        return None
    if q.answer_key in ANSWER_KEY_BLOCK_LIST:
        return f"Due to ANSWER_KEY_BLOCK_LIST (manual rejection)"
    if q.n_classes <= 1:
        return f"Since the number of unique labels is <= 1"
    if q.n_classes >= 15:
        return f"Due to suspected multi-binary"
    if spec.second_context_column is not None:
        return f"Double context column loading is not yet implemented."
    if (n_multiple_labels := sum(count > 1 for count in q.answer_counter_all().values())) <= 1:
    # if (n_multiple_labels := sum(count > 1 for count in q.answer_counter().values())) <= 1:
        return f"Only {n_multiple_labels} labels have multiple examples (require at least 2)."
    return None


def questions_from_spec(
        spec: QuestionSpec,
        verbose: bool = True,
) -> Tuple[List[specs.BaseDatasetSpec], List[specs.BaseDatasetSpec]]:
    context_key = headers[spec.context_column]
    accepted, rejected = [], []

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    def submit_question(q: specs.BaseDatasetSpec) -> None:
        reject_reason = get_question_reject_reason(spec, q)
        if reject_reason is not None:
            vprint(f"REJECT: Due to {reject_reason}, skipping {repr(q)[:500]}.")
            rejected.append(q)
        else:
            accepted.append(q)

    for i in spec.question_columns:
        answer_key = headers[i]
        if answer_key in ANSWER_KEY_BLOCK_LIST:
            vprint(f"MANUALLY-rejected (not included in count): {answer_key}")
            continue

        # if len(answer_set) >= 3 and any("," in ans for ans in answer_set):
        if answer_key in MULTI_BINARY_ANSWER_KEYS:
            q = specs.MultiBinaryDatasetSpec(
                id=f"{i}",
                context_key=context_key,
                answer_key=answer_key,
            )
        else:
            q = specs.MultipleChoiceDatasetSpec(
                id=f"{i}",
                context_key=context_key,
                answer_key=answer_key,
            )

        submit_question(q)
        # Add sub questions after original questions, if there are subquestions.
        if isinstance(q, specs.MultiBinaryDatasetSpec):
            for sub_q in q.make_subquestions():
                submit_question(sub_q)

    return accepted, rejected


if __name__ == "__main__":
    generate_specs(2)
