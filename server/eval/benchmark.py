"""Hand-written benchmark cases for the agent-routing pipeline.

Each case seeds a roster, marks a hot-cache ordering (most-recent last), and
specifies the single expected agent name the pipeline should surface. For
`chat` and `spawn` cases, expected_agent is None - we instead check that the
pipeline makes the right coarse decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


Expected = Literal["pick", "spawn", "chat"]


@dataclass
class BenchmarkCase:
    name: str
    roster: list[tuple[str, str]]  # (name, description)
    recently_used: list[str] = field(default_factory=list)   # most-recent last
    query: str = ""
    expected_kind: Expected = "pick"
    expected_agent: Optional[str] = None   # required when expected_kind == "pick"


# A deliberately noisy baseline roster that most cases share. ~25 agents of
# mixed domains so we can test precision under realistic overload.
_BASE_ROSTER: list[tuple[str, str]] = [
    ("Email to Alice", "Correspondence with Alice Chen about Q3 product launch"),
    ("Email to Bob", "Weekly standup follow-ups with Bob from engineering"),
    ("Email to Mom", "Keeping Mom updated on travel and life"),
    ("Vercel Job Offer", "Negotiating offer letter and start date with Vercel recruiting"),
    ("Stripe Interview Loop", "Scheduling and prepping for Stripe onsite interview rounds"),
    ("Tokyo Restaurant Search", "Finding dinner reservations for the November Tokyo trip"),
    ("Tokyo Hotel Booking", "Booking hotels in Shinjuku and Kyoto for the Japan trip"),
    ("Flight to SFO", "Booking JFK-SFO for the GM Summit conference"),
    ("Conference Agenda", "Which talks to attend at the GM Summit"),
    ("Q3 Budget Analysis", "Pulling quarterly spend numbers for finance review"),
    ("Q4 OKR Draft", "Drafting Q4 OKRs for team review"),
    ("Mom Birthday Reminder", "Nudge me three days before Mom's birthday Dec 2"),
    ("Dentist Appointment", "Schedule semi-annual dental checkup"),
    ("Doctor Physical", "Book annual physical with primary care"),
    ("Lease Renewal", "Reviewing apartment lease renewal terms"),
    ("Tax Documents", "Collecting W2 and 1099 docs for the CPA"),
    ("Credit Card Dispute", "Chasing a fraudulent charge on Chase card"),
    ("Insurance Quote", "Home insurance renewal shopping"),
    ("Gym Membership", "Compare pricing at nearby gyms"),
    ("Car Service Due", "Oil change by end of month"),
    ("Grocery List", "Weekly grocery shop"),
    ("Reading List", "Books to buy next"),
    ("Side Project Idea", "Brainstorming a weekend side project"),
    ("Resume Update", "Refreshing resume for networking"),
    ("Thank You Note - Sarah", "Follow-up thank-you after Sarah's coffee chat"),
]


def _with_base(extra: list[tuple[str, str]] = ()) -> list[tuple[str, str]]:
    return [*_BASE_ROSTER, *extra]


BENCHMARK: list[BenchmarkCase] = [
    # --- clean follow-ups: exact name match should win ---
    BenchmarkCase(
        name="follow_up_alice_launch",
        roster=_with_base(),
        recently_used=["Flight to SFO", "Email to Alice"],
        query="reply to Alice about the product launch - yes we're a go",
        expected_kind="pick",
        expected_agent="Email to Alice",
    ),
    BenchmarkCase(
        name="follow_up_vercel_offer",
        roster=_with_base(),
        recently_used=["Vercel Job Offer"],
        query="tell the Vercel recruiter I need another day to decide",
        expected_kind="pick",
        expected_agent="Vercel Job Offer",
    ),
    BenchmarkCase(
        name="follow_up_tokyo_dinner",
        roster=_with_base(),
        recently_used=["Tokyo Hotel Booking"],
        query="any update on the Tokyo dinner reservations?",
        expected_kind="pick",
        expected_agent="Tokyo Restaurant Search",
    ),

    # --- semantic follow-ups: name doesn't exact-match query ---
    BenchmarkCase(
        name="semantic_q3_finance",
        roster=_with_base(),
        recently_used=["Email to Bob"],
        query="pull the Q3 numbers for the finance review tomorrow",
        expected_kind="pick",
        expected_agent="Q3 Budget Analysis",
    ),
    BenchmarkCase(
        name="semantic_dentist",
        roster=_with_base(),
        recently_used=["Grocery List"],
        query="book my cleaning for next Thursday afternoon",
        expected_kind="pick",
        expected_agent="Dentist Appointment",
    ),
    BenchmarkCase(
        name="semantic_sfo_flight",
        roster=_with_base(),
        recently_used=["Conference Agenda"],
        query="any update on my travel for the summit?",
        expected_kind="pick",
        expected_agent="Flight to SFO",
    ),

    # --- disambiguation: multiple similar agents ---
    BenchmarkCase(
        name="disambig_email_which_alice",
        roster=_with_base([
            ("Email to Alice Wong", "Thread with Alice Wong from design team"),
        ]),
        recently_used=["Email to Alice"],
        query="follow up with Alice Chen about the launch",
        expected_kind="pick",
        expected_agent="Email to Alice",
    ),
    BenchmarkCase(
        name="disambig_mom_vs_birthday",
        roster=_with_base(),
        recently_used=["Email to Mom"],
        query="remind me three days before Mom's birthday",
        expected_kind="pick",
        expected_agent="Mom Birthday Reminder",
    ),

    # --- stale: agent is in roster but hasn't been used in a while ---
    BenchmarkCase(
        name="stale_tax_season",
        roster=_with_base(),
        recently_used=["Grocery List", "Gym Membership"],   # tax not recent
        query="did my CPA receive the 1099?",
        expected_kind="pick",
        expected_agent="Tax Documents",
    ),

    # --- net-new: nothing in the roster should match ---
    BenchmarkCase(
        name="new_pet_adoption",
        roster=_with_base(),
        recently_used=["Email to Bob"],
        query="look into adopting a cat from the local shelter",
        expected_kind="spawn",
        expected_agent=None,
    ),
    BenchmarkCase(
        name="new_language_class",
        roster=_with_base(),
        recently_used=["Resume Update"],
        query="find a mandarin tutor near me",
        expected_kind="spawn",
        expected_agent=None,
    ),

    # --- chat: no agent needed at all ---
    BenchmarkCase(
        name="chat_greeting",
        roster=_with_base(),
        recently_used=["Email to Alice"],
        query="hey, morning",
        expected_kind="chat",
        expected_agent=None,
    ),
    BenchmarkCase(
        name="chat_thanks",
        roster=_with_base(),
        recently_used=["Flight to SFO"],
        query="thanks, that's perfect",
        expected_kind="chat",
        expected_agent=None,
    ),
]
