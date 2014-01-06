# http://www.colourlovers.com/palette/1930/cheer_up_emo_kid
c = [
    '#556270',  # mighty slate
    '#4ECDC4',  # Pacificaa
    '#C7F464',  # apple chic
    '#FF6B6B',  # cherry pink
]


layout = {
    'yaxis': {
        'domain': [0, 0.3333333333333333],
        "title": "NIG  (Pa)",
        'gridcolor': 'white',
        "titlefont": {"color": c[0]},
        "tickfont": {"color": c[0]}},
    'yaxis2': {
        "overlaying": "y",
        "side": "right",
        "anchor": "x",
        'gridcolor': 'white',
        "title": "Normalized intensity  (a.u.)",
        "titlefont": {"color": c[0]},
        "tickfont": {"color": c[0]}},
    'plot_bgcolor': 'rgb(233,233,233)',
}
