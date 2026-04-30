"""Streamlit UI — upload screen → live interview → final evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

# Make `app.*` importable regardless of cwd when launched via `streamlit run app/streamlit_app.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.core.orchestrator import Orchestrator
from app.services.jd_parser import parse_jd
from app.services.resume_parser import parse_resume_bytes
from app.services.session_builder import (
    build_session,
    list_sessions,
    load_session,
    teardown_session,
)
from app.utils.async_runner import get_runner

st.set_page_config(page_title="Technical Interview Agent", layout="wide", page_icon=":speech_balloon:")


# ---------- Session bootstrap ----------

@st.cache_resource(show_spinner=False)
def _runner():
    return get_runner()


def _bootstrap() -> None:
    ss = st.session_state
    if "orch" not in ss:
        ss.orch = Orchestrator()
    ss.setdefault("setup_done", False)
    ss.setdefault("current_question", None)
    ss.setdefault("history", [])  # list[{role, content}]
    ss.setdefault("last_analysis", None)
    ss.setdefault("report", None)
    ss.setdefault("error", None)


_bootstrap()
runner = _runner()
orch: Orchestrator = st.session_state.orch


# ---------- Setup screen ----------

def render_setup() -> None:
    st.title("Technical Interview Agent")
    st.caption(
        "Upload a resume and paste a JD to build a new session graph, OR pick "
        "an existing session graph from Neo4j to start interviewing against it."
    )

    tab_new, tab_existing = st.tabs(["New interview", "Load existing session"])

    # --- New interview ---
    with tab_new:
        col_l, col_r = st.columns(2)
        with col_l:
            resume_file = st.file_uploader("Resume", type=["pdf", "docx", "txt"])
        with col_r:
            jd_text = st.text_area("Job description", height=300, placeholder="Paste the JD here…")

        can_start = resume_file is not None and jd_text.strip() != ""
        if st.button("Build graph & start interview", type="primary", disabled=not can_start):
            try:
                with st.spinner("Extracting resume + JD, building knowledge graph, ranking topics…"):
                    resume_text = parse_resume_bytes(resume_file.getvalue(), resume_file.name)
                    jd_clean = parse_jd(jd_text)
                    state = runner.run(build_session(resume_text, jd_clean))
                    orch.attach(state)
                st.session_state.setup_done = True
                st.rerun()
            except Exception as e:
                st.session_state.error = f"Setup failed: {e}"
                st.exception(e)

    # --- Load existing session ---
    with tab_existing:
        st.caption(
            "Pick a session that's already in Neo4j. The interview starts fresh "
            "(no prior Q&A is rehydrated), but uses the same Candidate, JD, and topics."
        )
        try:
            sessions = runner.run(list_sessions())
        except Exception as e:
            st.error(f"Failed to list sessions: {e}")
            sessions = []

        if not sessions:
            st.info("No sessions found in the graph yet — build one in the New interview tab.")
        else:
            options: dict[str, str] = {}
            for s in sessions:
                started = (s.get("started_at") or "")[:19]
                label = (
                    f"{s.get('candidate_name','Candidate')} · "
                    f"{s.get('jd_title','Role')} · "
                    f"{s.get('topic_count', 0)} topics · "
                    f"started {started}  (id: {s.get('id','')[:12]})"
                )
                options[label] = s.get("id", "")

            chosen_label = st.selectbox(
                "Available sessions", list(options.keys()), key="session_pick"
            )
            chosen_id = options.get(chosen_label, "")
            if st.button("Start interview from this session", type="primary", disabled=not chosen_id):
                try:
                    with st.spinner(f"Loading session {chosen_id[:12]}…"):
                        state = runner.run(load_session(chosen_id))
                        orch.attach(state)
                    st.session_state.setup_done = True
                    st.rerun()
                except Exception as e:
                    st.session_state.error = f"Load failed: {e}"
                    st.exception(e)


# ---------- Interview screen ----------

def render_sidebar() -> None:
    state = orch.state
    if state is None:
        return
    with st.sidebar:
        st.header("Topics")
        st.caption(f"Threshold to advance: {int(state.threshold * 100)}% coverage")
        for t in state.topics:
            icon = {"pending": "○", "active": "▶", "done": "✓", "skipped": "—"}[t.status]
            star = " ★" if t.must_have else ""
            src_tag = "  · _from resume_" if t.source == "resume" else ""
            st.markdown(f"**{icon} {t.name}**{star}{src_tag}")
            st.progress(min(t.coverage, 1.0))
            clar = f" · clar {t.clarification_count}" if t.clarification_count else ""
            st.caption(
                f"imp {t.importance:.1f} · attempts {t.attempts}{clar} · score {t.avg_score():.2f}"
            )
        st.divider()
        st.metric("Overall score", f"{state.overall_score():.2f}")
        st.caption(f"Session: `{state.session_id}`")
        if st.button("End & reset"):
            try:
                if orch.state is not None:
                    runner.run(teardown_session(orch.state.session_id))
            finally:
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()


def render_history() -> None:
    for h in st.session_state.history:
        with st.chat_message(h["role"]):
            st.markdown(h["content"])


def render_streaming_question() -> None:
    with st.chat_message("assistant"):
        placeholder = st.empty()
        chunks: list[str] = []
        for piece in runner.stream(orch.stream_next_question):
            chunks.append(piece)
            placeholder.markdown("".join(chunks) + "▌")
        full = "".join(chunks).strip()
        placeholder.markdown(full)
    st.session_state.current_question = full
    st.session_state.history.append({"role": "assistant", "content": full})


def render_analysis_panel() -> None:
    a = st.session_state.last_analysis
    if not a:
        return
    with st.expander("Last answer analysis", expanded=False):
        st.json(a)


def render_final_report() -> None:
    if st.session_state.report is None:
        with st.spinner("Generating final evaluation…"):
            st.session_state.report = runner.run(orch.finalize())
    r = st.session_state.report
    st.success("Interview complete.")
    c1, c2 = st.columns(2)
    c1.metric("Recommendation", r.hire_recommendation.replace("_", " ").title())
    c2.metric("Overall score", f"{r.overall_score:.2f}")
    st.markdown("### Summary")
    st.write(r.summary)
    if r.strengths:
        st.markdown("### Strengths")
        for s in r.strengths:
            st.markdown(f"- {s}")
    if r.weaknesses:
        st.markdown("### Weaknesses / gaps")
        for w in r.weaknesses:
            st.markdown(f"- {w}")
    if r.per_topic:
        st.markdown("### Per-topic")
        for tev in r.per_topic:
            with st.expander(f"{tev.topic} — {tev.score:.2f}"):
                st.write(tev.verdict)


def render_interview() -> None:
    state = orch.state
    if state is None:
        st.error("Interview not initialised.")
        return
    render_sidebar()
    st.title("Interview")
    if state.jd_title:
        st.caption(f"Role: {state.jd_title}")

    render_history()
    render_analysis_panel()

    if state.finished:
        render_final_report()
        return

    # Stream the next question if we don't have one queued
    if st.session_state.current_question is None:
        render_streaming_question()

    answer = st.chat_input("Your answer (be specific — explain trade-offs and internals)")
    if answer:
        st.session_state.history.append({"role": "user", "content": answer})
        try:
            with st.spinner("Analyzing…"):
                outcome = runner.run(
                    orch.submit_answer(st.session_state.current_question, answer)
                )
            analysis = outcome.analysis
            st.session_state.last_analysis = {
                "response_type": outcome.response_type,
                "attempt_consumed": outcome.attempt_consumed,
                "score": analysis.composite_score(),
                "correctness": analysis.correctness,
                "depth": analysis.depth,
                "relevance": analysis.relevance,
                "coverage_delta": analysis.coverage_delta,
                "contradiction": analysis.contradiction,
                "contradiction_evidence": analysis.contradiction_evidence,
                "evasive": analysis.evasive,
                "follow_up_hint": analysis.follow_up_hint,
            }
            # Always show any agent-generated text (clarifications, wrap-up on terminate).
            if outcome.clarification_text:
                st.session_state.history.append({
                    "role": "assistant",
                    "content": outcome.clarification_text,
                })
            if not outcome.keep_current_question:
                # Real answer / dodge / terminate: clear so next iteration either
                # streams a new question or renders the final evaluation.
                st.session_state.current_question = None
        except Exception as e:
            st.session_state.error = f"Answer processing failed: {e}"
            st.exception(e)
        st.rerun()


# ---------- Router ----------

if st.session_state.error:
    st.error(st.session_state.error)
    if st.button("Clear error"):
        st.session_state.error = None
        st.rerun()

if not st.session_state.setup_done:
    render_setup()
else:
    render_interview()
