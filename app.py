import base64
import io
import os

import streamlit as st
from datetime import datetime
from PIL import Image

from utils_simple import (
    process_file,
    analyze_from_text,
    analyze_image,
    generate_heatmap,
    save_analysis,
    get_latest_analyses,
    generate_report,
    search_pubmed,
    generate_statistics_report,  
    NCBI_EMAIL_CONFIGURED,
)
from chat_system import render_chat_interface, create_manual_chat_room
from report_qa_chat import ReportQAChat, ReportQASystem
from qa_interface import render_qa_chat_interface
from openrouter_client import (
    validate_api_key,
    OPENROUTER_MODEL,
    OPENROUTER_VISION_MODEL,
)



# Page config


st.set_page_config(
    page_title="Medical Image Analysis",
    page_icon="🧑‍⚕️",
    layout="wide",
)



# Session state initialisation

_DEFAULTS = {
    "openrouter_key": "",
    "file_data": None,
    "analysis_results": None,
    "file_name": None,
    "file_type": None,
    "OPENROUTER_API_KEY": None,
    "findings": [],
    "include_references": True,
    "api_key_valid": None,   # None = not checked, True = valid, False = invalid
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v



# Sidebar

with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input(
        "OpenRouter API Key",
        value=st.session_state.openrouter_key,
        type="password",
        help="Get your key at https://openrouter.ai/keys",
    )

    if api_key != st.session_state.openrouter_key:
        
        st.session_state.openrouter_key = api_key
        st.session_state.OPENROUTER_API_KEY = api_key if api_key else None
        st.session_state.api_key_valid = None  

    if api_key and st.session_state.api_key_valid is None:
        with st.spinner("Validating API key…"):
            ok, err = validate_api_key(api_key)
        st.session_state.api_key_valid = ok
        if ok:
            st.session_state.OPENROUTER_API_KEY = api_key
            
            if "qa_system" in st.session_state:
                st.session_state.qa_system.api_key = api_key
        else:
            st.session_state.OPENROUTER_API_KEY = None

    if st.session_state.api_key_valid is True:
        st.success("API key valid", icon="✅")
    elif st.session_state.api_key_valid is False:
        st.error("API key invalid — check your key and retry.")
    elif not api_key:
        st.info("Enter your OpenRouter API key to enable AI features.")

    
    if "qa_system" in st.session_state:
        if st.session_state.qa_system.api_key != st.session_state.OPENROUTER_API_KEY:
            st.session_state.qa_system.api_key = st.session_state.OPENROUTER_API_KEY

    st.subheader("Models")
    st.caption(f"Vision: `{OPENROUTER_VISION_MODEL}`")
    st.caption(f"Chat / Q&A: `{OPENROUTER_MODEL}`")

    if not NCBI_EMAIL_CONFIGURED:
        st.warning(
            "NCBI_EMAIL is not set. PubMed searches may be rate-limited. "
            "Add `NCBI_EMAIL` to your environment or `.streamlit/secrets.toml`.",
            icon="⚠️",
        )

    st.subheader("Analysis Options")
    enable_xai = st.checkbox("Enable Explainable AI (heatmap)", value=True)
    st.session_state.include_references = st.checkbox(
        "Include Medical References", value=st.session_state.include_references
    )

    st.subheader("Recent Analyses")
    for _a in get_latest_analyses(limit=5):
        st.caption(f"{_a.get('filename', 'Unknown')} — {_a.get('date', '')[:10]}")

    if st.button("Generate Statistics Report"):
        stats_report = generate_statistics_report()
        if stats_report:
            b64_pdf = base64.b64encode(stats_report.read()).decode()
            href = (
                f'<a href="data:application/pdf;base64,{b64_pdf}" '
                f'download="statistics_report.pdf">Download Statistics Report</a>'
            )
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No analyses yet to generate statistics from.")


# Title

st.title("Advanced Medical Image Analysis")
st.markdown(
    "Upload a medical image for AI-powered analysis and collaborate with colleagues."
)


# Main tabs

tab1, tab2, tab3, tab4 = st.tabs(
    ["Image Upload & Analysis", "Collaboration", "Report Q&A", "Reports"]
)


# ── Tab 1: Upload & Analyse

with tab1:
    uploaded_file = st.file_uploader(
        "Upload a medical image (JPEG, PNG, DICOM, NIfTI)",
        type=["jpg", "jpeg", "png", "dcm", "nii", "nii.gz"],
    )

    if uploaded_file:
        try:
            file_data = process_file(uploaded_file)

            if file_data:
                st.session_state.file_data = file_data
                st.session_state.file_name = uploaded_file.name
                st.session_state.file_type = file_data["type"]

                st.image(
                    file_data["data"],
                    caption=f"Uploaded {file_data['type'].upper()} image",
                    width=250
                )

                if enable_xai and st.checkbox("Show Explainability Heatmap", value=False):
                    overlay, heatmap = generate_heatmap(file_data["array"])
                    h_col1, h_col2 = st.columns(2)
                    with h_col1:
                        st.image(overlay, caption="Overlay", width=250)
                    with h_col2:
                        st.image(heatmap, caption="Heatmap", width=250)

                st.subheader("Analysis Method")
                analysis_mode = st.radio(
                    "Choose how to analyse the image",
                    [
                        "AI Vision Analysis (sends image to LLM)",
                        "Describe & Expand (type your observations)",
                    ],
                    horizontal=True,
                )

                user_findings = ""
                if analysis_mode.startswith("Describe"):
                    st.caption(
                        "Type what you observe — the AI will expand it into a full structured report."
                    )
                    user_findings = st.text_area(
                        "Your observations",
                        placeholder=(
                            "E.g.: Right lower lobe opacity, possible consolidation, "
                            "heart size normal, no pleural effusion."
                        ),
                        height=120,
                    )

                if not st.session_state.OPENROUTER_API_KEY:
                    st.warning("Please enter a valid OpenRouter API key in the sidebar.")
                elif st.button("Generate Analysis"):
                    if analysis_mode.startswith("Describe") and not user_findings.strip():
                        st.warning("Please describe your observations before generating the analysis.")
                    else:
                        with st.spinner("Generating analysis…"):
                            if analysis_mode.startswith("AI Vision"):
                                analysis_results = analyze_image(
                                    file_data["data"],
                                    st.session_state.OPENROUTER_API_KEY,
                                )
                            else:
                                analysis_results = analyze_from_text(
                                    user_findings,
                                    st.session_state.OPENROUTER_API_KEY,
                                )
                            analysis_results = save_analysis(
                                analysis_results, filename=uploaded_file.name
                            )

                        st.session_state.analysis_results = analysis_results
                        st.session_state.findings = analysis_results.get("findings", [])

                        st.subheader("Analysis Result")
                        st.markdown(analysis_results["analysis"])

                        if analysis_results.get("findings"):
                            st.subheader("Key Findings")
                            for idx, finding in enumerate(analysis_results["findings"], 1):
                                st.markdown(f"{idx}. {finding}")

                        if analysis_results.get("keywords"):
                            st.subheader("Keywords")
                            st.markdown(f"*{', '.join(analysis_results['keywords'])}*")

                        if st.session_state.include_references and analysis_results.get("keywords"):
                            st.subheader("Relevant Medical Literature")
                            references = search_pubmed(
                                analysis_results["keywords"], max_results=3
                            )
                            if references:
                                for ref in references:
                                    st.markdown(
                                        f"- **{ref['title']}**  \n"
                                        f"{ref['journal']}. {ref['year']} (PMID: {ref['id']})"
                                    )
                            else:
                                st.caption("No PubMed results found for these keywords.")

                        st.subheader("Report Generation")
                        pdf_buffer = generate_report(
                            analysis_results,
                            include_references=st.session_state.include_references,
                        )
                        b64_pdf = base64.b64encode(pdf_buffer.read()).decode()
                        date_str = datetime.now().strftime("%Y%m%d")
                        href = (
                            f'<a href="data:application/pdf;base64,{b64_pdf}" '
                            f'download="medical_report_{date_str}.pdf">Download PDF Report</a>'
                        )
                        st.markdown(href, unsafe_allow_html=True)

                        st.subheader("Collaborate")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("Start Case Discussion"):
                                desc = (
                                    analysis_results["findings"][0]
                                    if analysis_results.get("findings")
                                    else f"{uploaded_file.name} analysis"
                                )
                                created_case_id = create_manual_chat_room(
                                    "Dr. Anonymous", desc
                                )
                                st.session_state.current_case_id = created_case_id
                                st.rerun()
                        with col_b:
                            if st.button("Start Q&A Session"):
                                if "qa_chat" not in st.session_state:
                                    st.session_state.qa_chat = ReportQAChat()
                                room_name = f"Q&A for {uploaded_file.name}"
                                create_qa_id = st.session_state.qa_chat.create_qa_room(
                                    "Dr. Anonymous", room_name
                                )
                                st.session_state.current_qa_id = create_qa_id
                                st.rerun()
            else:
                st.error("Unable to process the uploaded file.")

        except Exception as exc:
            st.error(f"Error processing file: {exc}")

    elif st.session_state.analysis_results:
        st.subheader("Previous Analysis Results")
        prev = st.session_state.analysis_results
        st.markdown(prev["analysis"])

        if prev.get("findings"):
            st.subheader("Key Findings")
            for idx, finding in enumerate(prev["findings"], 1):
                st.markdown(f"{idx}. {finding}")

        st.subheader("Report")
        if st.button("Generate PDF Report"):
            pdf_buffer = generate_report(
                prev, include_references=st.session_state.include_references
            )
            b64_pdf = base64.b64encode(pdf_buffer.read()).decode()
            date_str = datetime.now().strftime("%Y%m%d")
            href = (
                f'<a href="data:application/pdf;base64,{b64_pdf}" '
                f'download="medical_report_{date_str}.pdf">Download PDF Report</a>'
            )
            st.markdown(href, unsafe_allow_html=True)


# ── Tab 2: Collaboration 

with tab2:
    try:
        render_chat_interface()
    except Exception as exc:
        st.error(f"Error in chat interface: {exc}")
        st.info("If you're trying to create a new discussion, upload and analyse an image first.")
        st.subheader("Create Discussion Without Image")
        manual_creator = st.text_input("Your Name", value="Dr. Anonymous")
        manual_description = st.text_input("Case Description")
        if st.button("Create Manual Discussion") and manual_description:
            case_id = create_manual_chat_room(manual_creator, manual_description)
            st.session_state.current_case_id = case_id
            st.rerun()


# ── Tab 3: Report Q&A

with tab3:
    render_qa_chat_interface()


# ── Tab 4: Reports & Analytics

with tab4:
    st.subheader("Medical Report & Analytics")
    st.markdown("### Analysis History")

    recent_analyses = get_latest_analyses(limit=10)

    if recent_analyses:
        for idx, analysis in enumerate(recent_analyses, 1):
            with st.expander(
                f"{idx}. {analysis.get('filename', 'Unknown')} — {analysis.get('date', '')[:10]}"
            ):
                st.markdown(analysis.get("analysis", "No analysis available"))

                if analysis.get("findings"):
                    st.markdown("**Key Findings:**")
                    for fidx, finding in enumerate(analysis["findings"], 1):
                        st.markdown(f"{fidx}. {finding}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Generate Report #{idx}", key=f"rpt_{idx}"):
                        pdf_buffer = generate_report(
                            analysis,
                            include_references=st.session_state.include_references,
                        )
                        b64_pdf = base64.b64encode(pdf_buffer.read()).decode()
                        report_name = f"report_{analysis.get('id', 'unknown')[:8]}.pdf"
                        href = (
                            f'<a href="data:application/pdf;base64,{b64_pdf}" '
                            f'download="{report_name}">Download Report</a>'
                        )
                        st.markdown(href, unsafe_allow_html=True)

                with col2:
                    if st.button(f"Q&A on Report #{idx}", key=f"qa_{idx}"):
                        if "qa_chat" not in st.session_state:
                            st.session_state.qa_chat = ReportQAChat()
                        room_name = f"Q&A for {analysis.get('filename', 'Unknown')}"
                        create_qa_id = st.session_state.qa_chat.create_qa_room(
                            "Dr. Anonymous", room_name
                        )
                        st.session_state.current_qa_id = create_qa_id
                        st.rerun()
    else:
        st.info("No previous analyses found. Upload and analyse an image to get started.")

    st.markdown("### Statistics")
    if st.button("Generate Comprehensive Statistics"):
        stats_report = generate_statistics_report()
        if stats_report:
            b64_pdf = base64.b64encode(stats_report.read()).decode()
            href = (
                f'<a href="data:application/pdf;base64,{b64_pdf}" '
                f'download="statistics_report.pdf">Download Statistics Report</a>'
            )
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No analyses yet to generate statistics from.")
