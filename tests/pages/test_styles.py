"""
Tests for the styles module.

This module contains tests for the styles.py module, which
provides the styling constants and dictionaries for the investment funnel dashboard.
"""

from funnel.pages.styles.styles import (
    SIDE_BAR_WIDTH,
    OPTION_WIDTH,
    OPTION_ELEMENT,
    OPTION_BTN,
    MAIN_TITLE,
    SUB_TITLE,
    DESCRIP_INFO,
    SIDEBAR_STYLE,
    NAV_BTN,
    GRAPH_LEFT,
    GRAPH_RIGHT,
    MOBILE_PAGE,
    LOADING_STYLE,
)


def test_style_constants() -> None:
    """
    Test that the style constants have the expected values.

    This test verifies that the width constants used for layout calculations
    have the correct percentage values.
    """
    assert SIDE_BAR_WIDTH == "16.67%", "SIDE_BAR_WIDTH should be 16.67%"
    assert OPTION_WIDTH == "20%", "OPTION_WIDTH should be 20%"


def test_option_element_style() -> None:
    """
    Test that the OPTION_ELEMENT dictionary has the expected keys and values.

    This test verifies that the style dictionary for option elements
    contains the correct margin and font-size properties.
    """
    assert "margin" in OPTION_ELEMENT, "OPTION_ELEMENT should have margin property"
    assert "font-size" in OPTION_ELEMENT, "OPTION_ELEMENT should have font-size property"
    assert OPTION_ELEMENT["margin"] == "1%", "OPTION_ELEMENT margin should be 1%"
    assert OPTION_ELEMENT["font-size"] == "12px", "OPTION_ELEMENT font-size should be 12px"


def test_option_button_style() -> None:
    """
    Test that the OPTION_BTN dictionary has the expected keys and values.

    This test verifies that the style dictionary for option buttons
    contains the correct properties for appearance and layout.
    """
    assert "margin" in OPTION_BTN, "OPTION_BTN should have margin property"
    assert "height" in OPTION_BTN, "OPTION_BTN should have height property"
    assert "background-color" in OPTION_BTN, "OPTION_BTN should have background-color property"
    assert "color" in OPTION_BTN, "OPTION_BTN should have color property"
    assert "font-size" in OPTION_BTN, "OPTION_BTN should have font-size property"
    assert "verticalAlign" in OPTION_BTN, "OPTION_BTN should have verticalAlign property"
    assert "border-radius" in OPTION_BTN, "OPTION_BTN should have border-radius property"

    assert OPTION_BTN["margin"] == "3%", "OPTION_BTN margin should be 3%"
    assert OPTION_BTN["height"] == "60px", "OPTION_BTN height should be 60px"
    assert OPTION_BTN["background-color"] == "#111723", "OPTION_BTN background-color should be #111723"
    assert OPTION_BTN["color"] == "white", "OPTION_BTN color should be white"
    assert OPTION_BTN["font-size"] == "12px", "OPTION_BTN font-size should be 12px"
    assert OPTION_BTN["verticalAlign"] == "middle", "OPTION_BTN verticalAlign should be middle"
    assert OPTION_BTN["border-radius"] == "15px", "OPTION_BTN border-radius should be 15px"


def test_title_styles() -> None:
    """
    Test that the title style dictionaries have the expected keys and values.

    This test verifies that the style dictionaries for main and sub titles
    contain the correct properties for text alignment, margins, and typography.
    """
    # Test MAIN_TITLE
    assert "text-align" in MAIN_TITLE, "MAIN_TITLE should have text-align property"
    assert "margin" in MAIN_TITLE, "MAIN_TITLE should have margin property"
    assert "margin-top" in MAIN_TITLE, "MAIN_TITLE should have margin-top property"
    assert "font-size" in MAIN_TITLE, "MAIN_TITLE should have font-size property"
    assert "font-weight" in MAIN_TITLE, "MAIN_TITLE should have font-weight property"

    assert MAIN_TITLE["text-align"] == "left", "MAIN_TITLE text-align should be left"
    assert MAIN_TITLE["margin"] == "2%", "MAIN_TITLE margin should be 2%"
    assert MAIN_TITLE["margin-top"] == "16px", "MAIN_TITLE margin-top should be 16px"
    assert MAIN_TITLE["font-size"] == "16px", "MAIN_TITLE font-size should be 16px"
    assert MAIN_TITLE["font-weight"] == "600", "MAIN_TITLE font-weight should be 600"

    # Test SUB_TITLE
    assert "text-align" in SUB_TITLE, "SUB_TITLE should have text-align property"
    assert "margin-top" in SUB_TITLE, "SUB_TITLE should have margin-top property"
    assert "margin-bottom" in SUB_TITLE, "SUB_TITLE should have margin-bottom property"
    assert "margin-left" in SUB_TITLE, "SUB_TITLE should have margin-left property"
    assert "font-size" in SUB_TITLE, "SUB_TITLE should have font-size property"
    assert "font-weight" in SUB_TITLE, "SUB_TITLE should have font-weight property"
    assert "color" in SUB_TITLE, "SUB_TITLE should have color property"

    assert SUB_TITLE["text-align"] == "left", "SUB_TITLE text-align should be left"
    assert SUB_TITLE["margin-top"] == "6%", "SUB_TITLE margin-top should be 6%"
    assert SUB_TITLE["margin-bottom"] == "1%", "SUB_TITLE margin-bottom should be 1%"
    assert SUB_TITLE["margin-left"] == "2%", "SUB_TITLE margin-left should be 2%"
    assert SUB_TITLE["font-size"] == "12px", "SUB_TITLE font-size should be 12px"
    assert SUB_TITLE["font-weight"] == "500", "SUB_TITLE font-weight should be 500"
    assert SUB_TITLE["color"] == "#191919", "SUB_TITLE color should be #191919"


def test_description_info_style() -> None:
    """
    Test that the DESCRIP_INFO dictionary has the expected keys and values.

    This test verifies that the style dictionary for description information
    contains the correct properties for text alignment, margins, and typography.
    """
    assert "text-align" in DESCRIP_INFO, "DESCRIP_INFO should have text-align property"
    assert "margin" in DESCRIP_INFO, "DESCRIP_INFO should have margin property"
    assert "font-size" in DESCRIP_INFO, "DESCRIP_INFO should have font-size property"
    assert "color" in DESCRIP_INFO, "DESCRIP_INFO should have color property"

    assert DESCRIP_INFO["text-align"] == "left", "DESCRIP_INFO text-align should be left"
    assert DESCRIP_INFO["margin"] == "2%", "DESCRIP_INFO margin should be 2%"
    assert DESCRIP_INFO["font-size"] == "12px", "DESCRIP_INFO font-size should be 12px"
    assert DESCRIP_INFO["color"] == "#5d5d5d", "DESCRIP_INFO color should be #5d5d5d"


def test_sidebar_style() -> None:
    """
    Test that the SIDEBAR_STYLE dictionary has the expected keys and values.

    This test verifies that the style dictionary for the sidebar
    contains the correct properties for positioning, dimensions, and appearance.
    """
    assert "position" in SIDEBAR_STYLE, "SIDEBAR_STYLE should have position property"
    assert "top" in SIDEBAR_STYLE, "SIDEBAR_STYLE should have top property"
    assert "left" in SIDEBAR_STYLE, "SIDEBAR_STYLE should have left property"
    assert "bottom" in SIDEBAR_STYLE, "SIDEBAR_STYLE should have bottom property"
    assert "width" in SIDEBAR_STYLE, "SIDEBAR_STYLE should have width property"
    assert "padding" in SIDEBAR_STYLE, "SIDEBAR_STYLE should have padding property"
    assert "background-color" in SIDEBAR_STYLE, "SIDEBAR_STYLE should have background-color property"
    assert "display" in SIDEBAR_STYLE, "SIDEBAR_STYLE should have display property"
    assert "flex-direction" in SIDEBAR_STYLE, "SIDEBAR_STYLE should have flex-direction property"
    assert "overflow" in SIDEBAR_STYLE, "SIDEBAR_STYLE should have overflow property"

    assert SIDEBAR_STYLE["position"] == "fixed", "SIDEBAR_STYLE position should be fixed"
    assert SIDEBAR_STYLE["top"] == 0, "SIDEBAR_STYLE top should be 0"
    assert SIDEBAR_STYLE["left"] == 0, "SIDEBAR_STYLE left should be 0"
    assert SIDEBAR_STYLE["bottom"] == 0, "SIDEBAR_STYLE bottom should be 0"
    assert SIDEBAR_STYLE["width"] == SIDE_BAR_WIDTH, "SIDEBAR_STYLE width should be SIDE_BAR_WIDTH"
    assert SIDEBAR_STYLE["padding"] == "1rem", "SIDEBAR_STYLE padding should be 1rem"
    assert SIDEBAR_STYLE["background-color"] == "#111723", "SIDEBAR_STYLE background-color should be #111723"
    assert SIDEBAR_STYLE["display"] == "flex", "SIDEBAR_STYLE display should be flex"
    assert SIDEBAR_STYLE["flex-direction"] == "column", "SIDEBAR_STYLE flex-direction should be column"
    assert SIDEBAR_STYLE["overflow"] == "auto", "SIDEBAR_STYLE overflow should be auto"


def test_nav_button_style() -> None:
    """
    Test that the NAV_BTN dictionary has the expected keys and values.

    This test verifies that the style dictionary for navigation buttons
    contains the correct properties.
    """
    assert "a:color" in NAV_BTN, "NAV_BTN should have a:color property"
    assert NAV_BTN["a:color"] == "white", "NAV_BTN a:color should be white"


def test_graph_left_style() -> None:
    """
    Test that the GRAPH_LEFT dictionary has the expected keys and values.

    This test verifies that the style dictionary for the left graph panel
    contains the correct properties for positioning, dimensions, and appearance.
    """
    assert "position" in GRAPH_LEFT, "GRAPH_LEFT should have position property"
    assert "left" in GRAPH_LEFT, "GRAPH_LEFT should have left property"
    assert "top" in GRAPH_LEFT, "GRAPH_LEFT should have top property"
    assert "width" in GRAPH_LEFT, "GRAPH_LEFT should have width property"
    assert "bottom" in GRAPH_LEFT, "GRAPH_LEFT should have bottom property"
    assert "background-color" in GRAPH_LEFT, "GRAPH_LEFT should have background-color property"
    assert "padding" in GRAPH_LEFT, "GRAPH_LEFT should have padding property"
    assert "display" in GRAPH_LEFT, "GRAPH_LEFT should have display property"
    assert "flex-direction" in GRAPH_LEFT, "GRAPH_LEFT should have flex-direction property"
    assert "overflow" in GRAPH_LEFT, "GRAPH_LEFT should have overflow property"

    assert GRAPH_LEFT["position"] == "fixed", "GRAPH_LEFT position should be fixed"
    assert GRAPH_LEFT["left"] == SIDE_BAR_WIDTH, "GRAPH_LEFT left should be SIDE_BAR_WIDTH"
    assert GRAPH_LEFT["top"] == 0, "GRAPH_LEFT top should be 0"
    assert GRAPH_LEFT["width"] == OPTION_WIDTH, "GRAPH_LEFT width should be OPTION_WIDTH"
    assert GRAPH_LEFT["bottom"] == "0%", "GRAPH_LEFT bottom should be 0%"
    assert GRAPH_LEFT["background-color"] == "#d4d5d6", "GRAPH_LEFT background-color should be #d4d5d6"
    assert GRAPH_LEFT["padding"] == "8px", "GRAPH_LEFT padding should be 8px"
    assert GRAPH_LEFT["display"] == "flex", "GRAPH_LEFT display should be flex"
    assert GRAPH_LEFT["flex-direction"] == "column", "GRAPH_LEFT flex-direction should be column"
    assert GRAPH_LEFT["overflow"] == "auto", "GRAPH_LEFT overflow should be auto"


def test_graph_right_style() -> None:
    """
    Test that the GRAPH_RIGHT dictionary has the expected keys and values.

    This test verifies that the style dictionary for the right graph panel
    contains the correct properties for positioning, dimensions, and appearance.
    """
    assert "position" in GRAPH_RIGHT, "GRAPH_RIGHT should have position property"
    assert "left" in GRAPH_RIGHT, "GRAPH_RIGHT should have left property"
    assert "right" in GRAPH_RIGHT, "GRAPH_RIGHT should have right property"
    assert "top" in GRAPH_RIGHT, "GRAPH_RIGHT should have top property"
    assert "bottom" in GRAPH_RIGHT, "GRAPH_RIGHT should have bottom property"
    assert "padding" in GRAPH_RIGHT, "GRAPH_RIGHT should have padding property"
    assert "display" in GRAPH_RIGHT, "GRAPH_RIGHT should have display property"
    assert "flex-direction" in GRAPH_RIGHT, "GRAPH_RIGHT should have flex-direction property"
    assert "overflow" in GRAPH_RIGHT, "GRAPH_RIGHT should have overflow property"

    assert GRAPH_RIGHT["position"] == "fixed", "GRAPH_RIGHT position should be fixed"
    assert GRAPH_RIGHT["left"] == "36.67%", "GRAPH_RIGHT left should be 36.67%"
    assert GRAPH_RIGHT["right"] == "0%", "GRAPH_RIGHT right should be 0%"
    assert GRAPH_RIGHT["top"] == 0, "GRAPH_RIGHT top should be 0"
    assert GRAPH_RIGHT["bottom"] == "0%", "GRAPH_RIGHT bottom should be 0%"
    assert GRAPH_RIGHT["padding"] == "4px", "GRAPH_RIGHT padding should be 4px"
    assert GRAPH_RIGHT["display"] == "flex", "GRAPH_RIGHT display should be flex"
    assert GRAPH_RIGHT["flex-direction"] == "column", "GRAPH_RIGHT flex-direction should be column"
    assert GRAPH_RIGHT["overflow"] == "auto", "GRAPH_RIGHT overflow should be auto"


def test_mobile_page_style() -> None:
    """
    Test that the MOBILE_PAGE dictionary has the expected keys and values.

    This test verifies that the style dictionary for the mobile page
    contains the correct properties for positioning, dimensions, and appearance.
    """
    assert "position" in MOBILE_PAGE, "MOBILE_PAGE should have position property"
    assert "padding" in MOBILE_PAGE, "MOBILE_PAGE should have padding property"
    assert "display" in MOBILE_PAGE, "MOBILE_PAGE should have display property"
    assert "flex-direction" in MOBILE_PAGE, "MOBILE_PAGE should have flex-direction property"
    assert "overflow" in MOBILE_PAGE, "MOBILE_PAGE should have overflow property"
    assert "background-color" in MOBILE_PAGE, "MOBILE_PAGE should have background-color property"
    assert "top" in MOBILE_PAGE, "MOBILE_PAGE should have top property"
    assert "left" in MOBILE_PAGE, "MOBILE_PAGE should have left property"
    assert "bottom" in MOBILE_PAGE, "MOBILE_PAGE should have bottom property"
    assert "width" in MOBILE_PAGE, "MOBILE_PAGE should have width property"

    assert MOBILE_PAGE["position"] == "fixed", "MOBILE_PAGE position should be fixed"
    assert MOBILE_PAGE["padding"] == "4px", "MOBILE_PAGE padding should be 4px"
    assert MOBILE_PAGE["display"] == "flex", "MOBILE_PAGE display should be flex"
    assert MOBILE_PAGE["flex-direction"] == "column", "MOBILE_PAGE flex-direction should be column"
    assert MOBILE_PAGE["overflow"] == "auto", "MOBILE_PAGE overflow should be auto"
    assert MOBILE_PAGE["background-color"] == "#111723", "MOBILE_PAGE background-color should be #111723"
    assert MOBILE_PAGE["top"] == 0, "MOBILE_PAGE top should be 0"
    assert MOBILE_PAGE["left"] == 0, "MOBILE_PAGE left should be 0"
    assert MOBILE_PAGE["bottom"] == 0, "MOBILE_PAGE bottom should be 0"
    assert MOBILE_PAGE["width"] == "100%", "MOBILE_PAGE width should be 100%"


def test_loading_style() -> None:
    """
    Test that the LOADING_STYLE dictionary has the expected keys and values.

    This test verifies that the style dictionary for the loading indicator
    contains the correct properties for positioning, dimensions, and appearance.
    """
    assert "background" in LOADING_STYLE, "LOADING_STYLE should have background property"
    assert "display" in LOADING_STYLE, "LOADING_STYLE should have display property"
    assert "justifyContent" in LOADING_STYLE, "LOADING_STYLE should have justifyContent property"
    assert "alignItems" in LOADING_STYLE, "LOADING_STYLE should have alignItems property"
    assert "position" in LOADING_STYLE, "LOADING_STYLE should have position property"
    assert "left" in LOADING_STYLE, "LOADING_STYLE should have left property"
    assert "right" in LOADING_STYLE, "LOADING_STYLE should have right property"
    assert "top" in LOADING_STYLE, "LOADING_STYLE should have top property"
    assert "bottom" in LOADING_STYLE, "LOADING_STYLE should have bottom property"

    assert LOADING_STYLE["background"] == "white", "LOADING_STYLE background should be white"
    assert LOADING_STYLE["display"] == "flex", "LOADING_STYLE display should be flex"
    assert LOADING_STYLE["justifyContent"] == "center", "LOADING_STYLE justifyContent should be center"
    assert LOADING_STYLE["alignItems"] == "center", "LOADING_STYLE alignItems should be center"
    assert LOADING_STYLE["position"] == "fixed", "LOADING_STYLE position should be fixed"
    assert LOADING_STYLE["left"] == "36.67%", "LOADING_STYLE left should be 36.67%"
    assert LOADING_STYLE["right"] == "0%", "LOADING_STYLE right should be 0%"
    assert LOADING_STYLE["top"] == 0, "LOADING_STYLE top should be 0"
    assert LOADING_STYLE["bottom"] == "0%", "LOADING_STYLE bottom should be 0%"
