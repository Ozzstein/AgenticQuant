"""Read-only research view sharing the authoritative application registry."""
import streamlit as st

from src.dashboard import research_page
from src.operations.reporting import research_view
from src.utils.config import get_config


def main():
    st.set_page_config(page_title='QuantAgentLab Research',layout='wide')
    research_page(research_view(get_config().operations.research_registry_path))


if __name__=='__main__':
    main()
