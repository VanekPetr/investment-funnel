"""
Tests for the nav module.

This module contains tests for the nav.py module, which
provides the navigation sidebar component for the investment funnel dashboard.
"""

import dash_bootstrap_components as dbc
from dash import html

from funnel.nav import get_navbar


def test_get_navbar() -> None:
    """
    Test that the get_navbar function returns a properly structured navbar component.

    This test verifies that the navbar component has the expected structure,
    including the correct width, style attributes, and children components.
    """
    # Get the navbar component
    navbar = get_navbar()

    # Test that it's a Bootstrap column with the correct width
    assert isinstance(navbar, dbc.Col), "Navbar should be a Bootstrap column"
    assert navbar.width == 2, "Navbar should have width=2"

    # Test the style attributes
    style = navbar.style
    assert style["backgroundColor"] == "#212529", "Navbar should have dark background color"
    assert style["height"] == "100vh", "Navbar should have full viewport height"
    assert style["position"] == "fixed", "Navbar should have fixed position"
    assert style["left"] == 0, "Navbar should be positioned at left=0"
    assert style["top"] == 0, "Navbar should be positioned at top=0"
    assert style["bottom"] == 0, "Navbar should be positioned at bottom=0"
    assert style["paddingTop"] == "1rem", "Navbar should have paddingTop=1rem"
    assert style["borderRight"] == "1px solid #343a40", "Navbar should have a dark border"

    # Test that it has the expected children
    children = navbar.children
    assert len(children) == 2, "Navbar should have 2 children"

    # Test the logo div
    logo_div = children[0]
    assert isinstance(logo_div, html.Div), "First child should be a Div"
    assert logo_div.style["textAlign"] == "center", "Logo div should have centered text"
    assert isinstance(logo_div.children, html.Img), "Logo div should contain an Img"
    assert logo_div.children.src == "/assets/logo.png", "Logo should have correct src"

    # Test the navigation links div
    nav_div = children[1]
    assert isinstance(nav_div, html.Div), "Second child should be a Div"
    assert nav_div.style["padding"] == "1rem", "Nav div should have padding=1rem"

    # Test the navigation component
    nav = nav_div.children[0]
    assert isinstance(nav, dbc.Nav), "Nav div should contain a Nav component"
    assert nav.vertical is True, "Nav should be vertical"
    assert nav.pills is True, "Nav should use pills style"

    # Test the navigation links
    nav_links = nav.children
    assert len(nav_links) == 4, "Nav should have 4 links"

    # Test each navigation link
    expected_links = [
        ("Overview", "/overview"),
        ("Lifecycle", "/lifecycle"),
        ("Backtest", "/backtest"),
        ("AI Feature Selection", "/ai_feature_selection")
    ]

    for i, (label, href) in enumerate(expected_links):
        link = nav_links[i]
        assert isinstance(link, dbc.NavLink), f"Link {i} should be a NavLink"
        assert link.children == label, f"Link {i} should have label '{label}'"
        assert link.href == href, f"Link {i} should have href '{href}'"
        assert link.active == "exact", f"Link {i} should have active='exact'"
