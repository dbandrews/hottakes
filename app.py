import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, callback, dcc, html

from hottakes.bot.inference import (
    BrowserServiceStrategy,
    generate_comments_modal,
    generate_comments_openai,
    get_article_details,
    slice_article_text,
)

# Sample URLs
SAMPLE_URLS = [
    "https://www.pinkbike.com/news/letter-from-the-editor-pinkbikes-next-chapter-with-outside.html",
    "https://www.pinkbike.com/news/british-councillor-objects-to-local-bike-park-due-to-fears-of-child-molesters-and-aggressive-males.html",
    "https://www.pinkbike.com/news/graham-agassiz-update-red-bull-rampage-kona-2016.html",
    "https://www.pinkbike.com/news/tokyo-2020-olympics-postponed.html",
    "https://www.pinkbike.com/news/trek-being-sued-for-5-million-over-wavecel-safety-claims.html",
    "https://www.pinkbike.com/news/mountain-biker-in-spain-shot-after-being-mistaken-for-a-rabbit.html",
    "https://www.pinkbike.com/news/pole-bicycles-ceo-resigns-company-founder-leo-kokkonen-to-take-leading-role.html",
    "https://www.pinkbike.com/news/yt-industries-acquired-by-private-equity-group-ardian.html",
    "https://www.pinkbike.com/news/what-we-know-so-far-about-the-heartbreaking-murder-of-moriah-mo-wilson.html",
    "https://www.pinkbike.com/news/brage-vestavik-joins-red-bull.html",
    "https://www.pinkbike.com/news/jared-graves-diagnosed-with-a-brain-tumor.html",
    "https://www.pinkbike.com/news/paul-bas-this-is-not-an-injury-update.html",
    "https://www.pinkbike.com/news/judges-results-x-games-real-mtb.html",
    "https://www.pinkbike.com/news/spengle-3-spoke-carbon-wheels-video-2019.html",
    "https://www.pinkbike.com/news/first-look-shimanos-new-deore-12-speed-group-plus-other-2021-updates.html",
    "https://www.pinkbike.com/news/the-top-20-pinkbike-comments-of-the-past-decade.html",
    "https://www.pinkbike.com/news/uci-and-warner-bros-discovery-announce-viewing-options-and-new-world-series-branding-for-2023-world-cups.html",
    "https://www.pinkbike.com/news/field-test-2020-pole-stamina-140-the-fastest-trail-bike.html",
    "https://www.pinkbike.com/news/adolf-silva-without-a-bike-sponsor-for-2021.html",
    "https://www.pinkbike.com/news/video-jaxson-riddles-50-foot-huck-to-flat-crash.html",
]

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    [
        html.H1("Hottakes Comment Generator", className="my-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id="url-dropdown",
                            options=[{"label": url, "value": url} for url in SAMPLE_URLS],
                            value=SAMPLE_URLS[0],
                            className="mb-3",
                        ),
                        dbc.Button("Generate Comments", id="submit-button", color="primary", className="mb-4"),
                    ]
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Article Details", className="mb-3"),
                        dcc.Loading(
                            id="loading-article",
                            children=[
                                html.Div(id="article-title", className="mb-2"),
                                html.Div(id="article-text", className="mb-4"),
                            ],
                            type="default",
                        ),
                    ]
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Original Top Comment"),
                        dcc.Loading(
                            id="loading-top-comment",
                            children=[
                                html.Div(id="top-comment", className="p-3 border rounded mb-4"),
                            ],
                            type="default",
                        ),
                    ]
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("GPT-4o-mini Comment"),
                        html.H6("Sampling Parameters", className="mb-3"),
                        dbc.Label("Number of Shots (Examples)"),
                        dcc.Dropdown(
                            id="gpt-shots-dropdown",
                            options=[
                                {"label": "3 shots", "value": 3},
                                {"label": "5 shots", "value": 5},
                                {"label": "10 shots", "value": 10},
                                {"label": "100 shots", "value": 100},
                            ],
                            value=3,
                            className="mb-3",
                        ),
                        dbc.Label("Temperature"),
                        dcc.Slider(
                            id="gpt-temperature-slider",
                            min=0,
                            max=2,
                            step=0.01,
                            value=1.0,
                            marks={0: "0", 1: "1", 2: "2"},
                            className="mb-2",
                        ),
                        html.Div(id="gpt-temperature-value", className="mb-3"),
                        dbc.Label("Top P"),
                        dcc.Slider(
                            id="gpt-top-p-slider",
                            min=0,
                            max=1,
                            step=0.01,
                            value=1.0,
                            marks={0: "0", 1: "1"},
                            className="mb-2",
                        ),
                        html.Div(id="gpt-top-p-value", className="mb-4"),
                        dcc.Loading(
                            id="loading-gpt4",
                            children=[
                                html.Div(id="gpt4-output", className="p-3 border rounded mb-2"),
                                dbc.Collapse(
                                    dbc.Card(dbc.CardBody(html.Pre(id="gpt4-prompt", className="small"))),
                                    id="gpt4-prompt-collapse",
                                ),
                                dbc.Button(
                                    "Show Prompt",
                                    id="gpt4-prompt-button",
                                    className="mt-2",
                                    color="secondary",
                                    size="sm",
                                ),
                            ],
                            type="default",
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.H4("Hottakes Comment"),
                        html.H6("Sampling Parameters", className="mb-3"),
                        dbc.Label("Temperature"),
                        dcc.Slider(
                            id="modal-temperature-slider",
                            min=0,
                            max=2,
                            step=0.01,
                            value=1.0,
                            marks={0: "0", 1: "1", 2: "2"},
                            className="mb-2",
                        ),
                        html.Div(id="modal-temperature-value", className="mb-3"),
                        dbc.Label("Top P"),
                        dcc.Slider(
                            id="modal-top-p-slider",
                            min=0,
                            max=1,
                            step=0.01,
                            value=1.0,
                            marks={0: "0", 1: "1"},
                            className="mb-2",
                        ),
                        html.Div(id="modal-top-p-value", className="mb-4"),
                        dcc.Loading(
                            id="loading-modal",
                            children=[
                                html.Div(id="modal-output", className="p-3 border rounded mb-2"),
                                dbc.Collapse(
                                    dbc.Card(dbc.CardBody(html.Pre(id="modal-prompt", className="small"))),
                                    id="modal-prompt-collapse",
                                ),
                                dbc.Button(
                                    "Show Prompt",
                                    id="modal-prompt-button",
                                    className="mt-2",
                                    color="secondary",
                                    size="sm",
                                ),
                            ],
                            type="default",
                        ),
                    ],
                    width=6,
                ),
            ]
        ),
    ]
)


# Add collapse toggle callbacks
@callback(
    Output("modal-prompt-collapse", "is_open"),
    Input("modal-prompt-button", "n_clicks"),
    State("modal-prompt-collapse", "is_open"),
)
def toggle_modal_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(
    Output("gpt4-prompt-collapse", "is_open"),
    Input("gpt4-prompt-button", "n_clicks"),
    State("gpt4-prompt-collapse", "is_open"),
)
def toggle_gpt4_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(
    [
        Output("article-title", "children"),
        Output("article-text", "children"),
        Output("top-comment", "children"),
        Output("gpt4-output", "children"),
        Output("gpt4-prompt", "children"),
        Output("modal-output", "children"),
        Output("modal-prompt", "children"),
    ],
    Input("submit-button", "n_clicks"),
    State("url-dropdown", "value"),
    State("modal-temperature-slider", "value"),
    State("modal-top-p-slider", "value"),
    State("gpt-temperature-slider", "value"),
    State("gpt-top-p-slider", "value"),
    State("gpt-shots-dropdown", "value"),
    prevent_initial_call=True,
)
def update_output(n_clicks, url, modal_temp, modal_top_p, gpt_temp, gpt_top_p, num_shots):
    if not url:
        return "No URL selected.", "", "", "", "", "", ""

    # Get article details
    article = get_article_details(url, strategy=BrowserServiceStrategy())
    title_article_text = slice_article_text(f"{article['title']} {article['article_text']}", num_words=300)

    # Generate comments
    openai_comment, openai_prompt = generate_comments_openai(
        [title_article_text], num_shots=num_shots, model_id="gpt-4o-mini", temperature=gpt_temp, top_p=gpt_top_p
    )
    modal_comment, modal_prompt = generate_comments_modal(
        [title_article_text], temperature=modal_temp, top_p=modal_top_p
    )

    return (
        html.H5(article["title"]),
        [html.P(line) for line in title_article_text.split("\n") if line],
        html.P(article["top_comment_text"]),
        html.P(openai_comment[0]),
        openai_prompt[0],
        html.P(modal_comment[0]),
        modal_prompt[0],
    )


# Add slider value display callbacks
@callback(
    [
        Output("gpt-temperature-value", "children"),
        Output("gpt-top-p-value", "children"),
        Output("modal-temperature-value", "children"),
        Output("modal-top-p-value", "children"),
    ],
    [
        Input("gpt-temperature-slider", "value"),
        Input("gpt-top-p-slider", "value"),
        Input("modal-temperature-slider", "value"),
        Input("modal-top-p-slider", "value"),
    ],
)
def update_slider_values(gpt_temp, gpt_top_p, modal_temp, modal_top_p):
    return [
        f"Value: {gpt_temp:.2f}",
        f"Value: {gpt_top_p:.2f}",
        f"Value: {modal_temp:.2f}",
        f"Value: {modal_top_p:.2f}",
    ]


if __name__ == "__main__":
    app.run(debug=True)
