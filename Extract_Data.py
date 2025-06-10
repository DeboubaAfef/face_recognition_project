from icrawler.builtin import GoogleImageCrawler

names = [
    "Emma Watson",
    "Leonardo DiCaprio",
    "Scarlett Johansson",
    "Will Smith",
    "Angelina Jolie"
]

for name in names:
    crawler = GoogleImageCrawler(storage={"root_dir": f"raw/{name.replace(' ', '_')}"})
    crawler.crawl(keyword=name, max_num=50)
