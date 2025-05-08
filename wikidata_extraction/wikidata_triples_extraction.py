import requests
import pandas as pd
from tqdm import tqdm
import time
import os
from typing import List, Dict, Any, Tuple
from collections import defaultdict

class WikidataProcessor:
    def __init__(self, batch_size: int = 50, rate_limit_delay: float = 0.1, output_file: str = 'wikidata_statements.csv'):
        self.base_url = "https://www.wikidata.org/w/api.php"
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        self.label_cache = {}
        self.output_file = output_file
        
    # [Previous methods remain unchanged up to process_entities]
    def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Make API request with rate limiting"""
        time.sleep(self.rate_limit_delay)
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()

    def _format_value(self, datavalue: Dict[str, Any]) -> str:
        """Format value based on its datatype"""
        value_type = datavalue.get('type')
        value = datavalue.get('value')
        
        if value_type == 'wikibase-entityid':
            return value.get('id', '')
        elif value_type == 'time':
            return value.get('time', '')
        elif value_type == 'quantity':
            return value.get('unit', '')
        elif value_type == 'monolingualtext':
            return value.get('text', '')
        elif value_type == 'string':
            return value
        else:
            return str(value)

    def get_label(self, id_: str) -> str:
        """Get cached label or ID if not found"""
        return self.label_cache.get(id_, id_)

    def get_property_and_value_labels(self, prop: str, datavalue: Dict[str, Any]) -> Tuple[str, str]:
        """Get property label and format value appropriately"""
        prop_label = self.label_cache.get(prop, prop)
        
        value_type = datavalue.get('type')
        value = datavalue.get('value')
        
        if value_type == 'wikibase-entityid':
            entity_id = value.get('id')
            return prop_label, self.label_cache.get(entity_id, entity_id)
        
        return prop_label, self._format_value(datavalue)

    def get_entities_batch(self, qids: List[str]) -> Dict[str, Any]:
        """Fetch multiple entities in a single request"""
        params = {
            "action": "wbgetentities",
            "ids": "|".join(qids),
            "format": "json",
            "languages": "en",
            "props": "claims|labels"
        }
        return self._make_request(params)

    def get_labels_batch(self, ids: List[str]) -> None:
        """Fetch and cache labels for multiple IDs in a single request"""
        ids_to_fetch = [id_ for id_ in ids if id_ and id_ not in self.label_cache]
        
        if not ids_to_fetch:
            return
            
        max_batch = 50
        for i in range(0, len(ids_to_fetch), max_batch):
            batch = ids_to_fetch[i:i + max_batch]
            params = {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "format": "json",
                "languages": "en",
                "props": "labels"
            }
            
            try:
                data = self._make_request(params)
                for entity_id, entity_data in data.get('entities', {}).items():
                    label = entity_data.get('labels', {}).get('en', {}).get('value', entity_id)
                    self.label_cache[entity_id] = label
            except Exception as e:
                print(f"Error fetching labels batch: {str(e)}")
                for id_ in batch:
                    self.label_cache[id_] = id_

    def save_batch_results(self, statements: List[Dict], batch_num: int) -> None:
        """Save batch results to CSV file"""
        df = pd.DataFrame(statements)
        
        # If it's the first batch and the file doesn't exist, create it with headers
        if batch_num == 0 and not os.path.exists(self.output_file):
            df.to_csv(self.output_file, index=False)
        else:
            # Append without headers for subsequent batches
            df.to_csv(self.output_file, mode='a', header=False, index=False)
            
        print(f"Saved batch {batch_num} ({len(statements)} statements) to {self.output_file}")

    def process_entities(self, input_file: str) -> None:
        """Process entities from input file in optimized batches with incremental saving"""
        # Read and clean Q-IDs
        with open(input_file, 'r') as f:
            qids = [qid.strip() for line in f for qid in line.split('|') if qid.strip()]
        
        # Process entities in batches
        for batch_num in tqdm(range(0, len(qids), self.batch_size), desc="Processing entities"):
            batch_statements = []
            property_ids = set()
            entity_value_ids = set()
            
            batch_qids = qids[batch_num:batch_num + self.batch_size]
            try:
                batch_data = self.get_entities_batch(batch_qids)
                
                # First pass: collect property and entity value IDs
                for entity_id, entity_data in batch_data.get('entities', {}).items():
                    if 'labels' not in entity_data or 'en' not in entity_data['labels']:
                        continue
                        
                    subject_label = entity_data['labels']['en']['value']
                    self.label_cache[entity_id] = subject_label
                    
                    for prop, claims in entity_data.get('claims', {}).items():
                        property_ids.add(prop)
                        
                        for claim in claims:
                            if ('mainsnak' not in claim or 
                                claim['mainsnak'].get("datatype") == "external-id" or
                                'datavalue' not in claim['mainsnak']):
                                continue
                                
                            datavalue = claim['mainsnak']['datavalue']
                            if datavalue['type'] == 'wikibase-entityid':
                                entity_value_ids.add(datavalue['value']['id'])
                
                # Fetch labels for properties and entity values
                if property_ids:
                    self.get_labels_batch(list(property_ids))
                if entity_value_ids:
                    self.get_labels_batch(list(entity_value_ids))
                
                # Second pass: process statements with cached labels
                for entity_id, entity_data in batch_data.get('entities', {}).items():
                    if 'labels' not in entity_data or 'en' not in entity_data['labels']:
                        continue
                        
                    subject_label = self.get_label(entity_id)
                    
                    for prop, claims in entity_data.get('claims', {}).items():
                        for claim in claims:
                            if ('mainsnak' not in claim or 
                                claim['mainsnak'].get("datatype") == "external-id" or
                                'datavalue' not in claim['mainsnak']):
                                continue
                                
                            datavalue = claim['mainsnak']['datavalue']
                            prop_label, object_label = self.get_property_and_value_labels(prop, datavalue)
                            rank = claim.get('rank', 'normal')
                                
                            batch_statements.append({
                                'id': entity_id,
                                'subject': subject_label,
                                'predicate': prop_label,
                                'object': object_label,
                                'rank': rank
                            })
                
                # Save batch results and clear batch statements
                if batch_statements:
                    self.save_batch_results(batch_statements, batch_num // self.batch_size)
                batch_statements.clear()
                
            except Exception as e:
                print(f"Error processing batch {batch_num}-{batch_num+self.batch_size}: {str(e)}")
                continue
            
            # Clear sets for next batch
            property_ids.clear()
            entity_value_ids.clear()
        
        print(f"Completed processing {len(qids)} entities. Results saved to {self.output_file}")

if __name__ == "__main__":
    processor = WikidataProcessor(
        batch_size=50, 
        rate_limit_delay=0.1,
        output_file="output/wikidata_statements.csv"
    )
    processor.process_entities("input/keys.txt")