import pandas as pd

def scrape_sports_reference_player_stats(url):
    tables = pd.read_html(url)
    totals_table = tables[2]
    return totals_table

url = 'https://www.sports-reference.com/cbb/players/trey-galloway-1.html'
stats = scrape_sports_reference_player_stats(url)
print(stats)
