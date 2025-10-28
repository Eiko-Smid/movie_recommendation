# utils/ui_table.py
from __future__ import annotations
from math import ceil
from typing import Iterable, Optional, Union

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator


def render_table(
    df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    cols: Optional[Iterable[str]] = None,
    editable: bool = False,
    paginate: bool = True,
    page_size: int = 25,
    height: Optional[int] = 360,
    use_container_width: bool = True,
    key: Optional[str] = None,
    container: Optional[DeltaGenerator] = None,
) -> pd.DataFrame:
    """
    Render a (optionally paginated) table on a Streamlit page.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to display.
    title : str, optional
        Optional section title above the table.
    n_rows : int, optional
        If given, limit to the first `n_rows` (applied after pagination slice).
    n_cols : int, optional
        If given and `cols` is None, limit to the first `n_cols` columns.
    cols : Iterable[str], optional
        Explicit subset of columns to include (overrides n_cols).
    editable : bool, default False
        If True, uses st.data_editor and returns edited DataFrame.
    paginate : bool, default True
        If True, show simple pagination controls.
    page_size : int, default 25
        Rows per page when paginate=True.
    height : int, optional
        Table height in pixels (None lets Streamlit auto-size).
    use_container_width : bool, default True
        Make the table stretch to the parent container width.
    key : str, optional
        Streamlit widget key prefix (important to avoid clashes across pages).
    container : st.container/column/expander, optional
        If provided, renders inside this container (lets you control “position”).
        If None, renders at the current location.

    Returns
    -------
    pd.DataFrame
        The dataframe that was actually rendered (edited if `editable=True`).
    """
    host: DeltaGenerator = container if container is not None else st

    # Column subsetting
    _df = df
    if cols is not None:
        # Keep only requested columns that exist
        keep = [c for c in cols if c in _df.columns]
        _df = _df.loc[:, keep]
    elif n_cols is not None:
        _df = _df.iloc[:, : max(0, n_cols)]

    # Pagination slice
    page_key = f"{key or 'table'}_page"
    if paginate:
        total = len(_df)
        pages = max(1, ceil(total / page_size))
        # Place controls inline with the title for nice UX
        header_row = host.columns([1, 1, 2]) if pages > 1 else [host]
        if title:
            header_row[0].markdown(f"### {title}")
        if pages > 1:
            with header_row[1]:
                page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=pages,
                    value=st.session_state.get(page_key, 1),
                    step=1,
                    key=f"{page_key}_input",
                )
                st.session_state[page_key] = page
            start = (st.session_state[page_key] - 1) * page_size
            stop = start + page_size
            view = _df.iloc[start:stop]
        else:
            view = _df
    else:
        if title:
            host.markdown(f"### {title}")
        view = _df

    # Row limit (after pagination)
    if n_rows is not None:
        view = view.head(n_rows)

    # Render
    if editable:
        rendered = host.data_editor(
            view,
            use_container_width=use_container_width,
            height=height,
            num_rows="dynamic",
            key=f"{key or 'table'}_editor",
        )
        return rendered
    else:
        host.dataframe(
            view,
            use_container_width=use_container_width,
            height=height,
        )
        return view
