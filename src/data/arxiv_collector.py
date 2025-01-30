# src/data/arxiv_collector.py
import arxiv
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict

class ArxivCollector:
    def __init__(self):
        self.client = arxiv.Client()

    def collect_recent_papers(self, 
                            categories: List[str] = ['cs.AI', 'cs.LG', 'cs.CL'],
                            days_back: int = 30) -> List[Dict]:
        """
        Collect papers from specified categories published in last n days
        """
        papers = []
        search = arxiv.Search(
            query=f"cat:({' OR '.join(categories)})",
            max_results=100,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        for result in self.client.results(search):
            # Convert to dict for easier storage
            paper_dict = {
                'title': result.title,
                'abstract': result.summary,
                'authors': [author.name for author in result.authors],
                'categories': result.categories,
                'published': result.published,
                'updated': result.updated,
                'pdf_url': result.pdf_url,
                'arxiv_id': result.entry_id.split('/')[-1]
            }
            papers.append(paper_dict)

        return papers

    def save_to_csv(self, papers: List[Dict], filename: str):
        """
        Save collected papers to CSV
        """
        df = pd.DataFrame(papers)
        df.to_csv(f'data/raw/{filename}', index=False)
        print(f"Saved {len(papers)} papers to {filename}")

if __name__ == "__main__":
    # Test the collector
    collector = ArxivCollector()
    papers = collector.collect_recent_papers()
    collector.save_to_csv(papers, f'arxiv_papers_{datetime.now().strftime("%Y%m%d")}.csv')