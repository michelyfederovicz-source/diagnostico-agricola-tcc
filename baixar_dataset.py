from icrawler.builtin import BingImageCrawler

classes = {
    "Planta_saudavel": "healthy plant leaf",
    "Pulgoes": "aphids on plant leaf",
    "Lagartas": "caterpillar on plant leaf",
    "Acaros": "plant mites leaf",
    "Estresse_hidrico": "plant drought stress leaf",
    "Formiga_cortadeira": "leaf cutter ant plant"
}

for pasta, busca in classes.items():
    print(f"Baixando imagens para: {pasta}")

    crawler = BingImageCrawler(storage={'root_dir': f'dataset/{pasta}'})
    crawler.crawl(
        keyword=busca,
        max_num=200
    )