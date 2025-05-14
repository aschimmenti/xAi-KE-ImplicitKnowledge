import pandas as pd
from openai import OpenAI
import requests
import time
from typing import Dict, Optional
import json

property_questions = {
    "educated at": "Where did {entity} receive their education?",
    "instrument": "What instrument does {entity} play?",
    "place of birth": "Where was {entity} born?",
    "religion or worldview": "What is {entity}'s religious belief or worldview?",
    "child": "Who is {entity}'s child?",
    "member of political party": "Which political party does {entity} belong to?",
    "represented by": "Who represents {entity}?",
    "field of work": "In which field does {entity} work?",
    "occupation": "What does {entity} do for a living?",
    "academic degree": "What academic degree does {entity} hold?",
    "spouse": "Who is {entity} married to?",
    "military branch": "In which branch of the military did {entity} serve?",
    "member of": "Which organization is {entity} a member of?",
    "sibling": "Who is {entity}'s sibling?",
    "manner of death": "How did {entity} pass away?",
    "blood type": "What is {entity}'s blood type?",
    "participant in": "What event did {entity} participate in?",
    "medical condition": "What medical condition does {entity} have?",
    "floruit": "During which period was {entity} most active or influential?",
    "place of burial": "Where is {entity} buried?",
    "influenced by": "Who influenced {entity}?",
    "position played on team / speciality": "What position or specialty does {entity} have on the team?",
    "sexual orientation": "What is {entity}'s sexual orientation?",
    "ancestral home": "Where is {entity}'s ancestral home?",
    "birth name": "What was {entity}'s name at birth?",
    "date of birth": "When was {entity} born?",
    "ethnic group": "What is {entity}'s ethnicity?",
    "notable work": "What is {entity}'s most famous work?",
    "noble title": "What noble title does {entity} hold?",
    "significant event": "What significant event marked {entity}'s life?",
    "cause of death": "What caused {entity}'s death?",
    "date of death": "When did {entity} pass away?",
    "student of": "Who taught {entity}?",
    "father": "Who is {entity}'s father?",
    "residence": "Where does {entity} live?",
    "mother": "Who is {entity}'s mother?",
    "supported sports team": "Which sports team does {entity} support?",
    "family name": "What is {entity}'s family name?",
    "position held": "What position does {entity} hold?",
    "member of sports team": "Which sports team is {entity} a member of?",
    "award received": "What award has {entity} received?",
    "employer": "Who employs {entity}?",
    "sport": "What sport does {entity} play?",
    "work location": "Where does {entity} work?",
    "married name": "What is {entity}'s married name?",
    "place of death": "Where did {entity} pass away?",
    "country of citizenship": "What country is {entity} a citizen of?"
}

def get_entity_label(qid: str) -> Optional[str]:
    """Fetch entity label from Wikidata using SPARQL."""
    url = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT ?label WHERE {{
      wd:{qid} rdfs:label ?label .
      FILTER(LANG(?label) = "en")
    }}
    """
    headers = {
        'User-Agent': 'QuestionGeneratorBot/1.0',
        'Accept': 'application/json'
    }
    try:
        response = requests.get(
            url, 
            params={'query': query, 'format': 'json'},
            headers=headers
        )
        response.raise_for_status()
        results = response.json()
        if results['results']['bindings']:
            return results['results']['bindings'][0]['label']['value']
        return None
    except Exception as e:
        print(f"Error fetching Wikidata label for {qid}: {e}")
        return None

class PromptProcessor:
    def __init__(self, property_questions: Dict[str, str], batch_size: int = 10):
        self.client = OpenAI()
        self.property_questions = property_questions
        self.batch_size = batch_size

    def generate_prompt(self, generated_text: str, entity_name: str, property_name: str) -> str:
        question = self.property_questions[property_name].format(entity=entity_name)
        return (f"You are a question answering expert. The following text contains information about an entity. "
                f"Read the question carefully and return the answer. If you do not find an answer, insert 'null'. "
                f"Return the value of the answer inside the JSON schema. The property name is {property_name}\n\n"
                f"{generated_text}\n\nQuestion:\n{question}")

    def process_prompt(self, prompt: str, property_name: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                    "name": "simple_string_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                        "content": {
                            "type": "string",
                            "property_name": property_name,
                            "description": "value of the property"
                        }
                        },
                        "required": [
                        "content"
                        ],
                        "additionalProperties": False
                    }
                    }
                },
                temperature=0.0,
                max_completion_tokens=3000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
                )
            # Estrai il valore 'content' dal JSON di risposta
            response_json = response.choices[0].message.content
            parsed_response = json.loads(response_json)
            return parsed_response['content']
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return f"Error: {str(e)}"

    def process_csv(self, input_file: str, output_file: str):
        # Read the CSV
        df = pd.read_csv(input_file)
        results = []

        for index, row in df.iterrows():
            try:
                # Extract Wikidata ID (assuming it's in the id column)
                qid = row['id']
                
                # Get entity name from Wikidata
                entity_name = get_entity_label(qid)
                if not entity_name:
                    print(f"Couldn't fetch entity name for {qid}, skipping...")
                    continue

                # Generate prompt
                prompt = self.generate_prompt(
                    row['generated_text'],
                    entity_name,
                    row['selected_property']
                )

                # Process with OpenAI
                response = self.process_prompt(prompt, row['selected_property'])

                # Store results
                results.append({
                    'id': qid,
                    'entity_name': entity_name,
                    'original_text': row['generated_text'],
                    'question': self.property_questions[row['selected_property']].format(entity=entity_name),
                    'answer': response
                })

                # Save progress after each batch
                if len(results) % self.batch_size == 0:
                    pd.DataFrame(results).to_csv(output_file, index=False)
                    print(f"Processed {len(results)} entries...")

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"Error processing row {index}: {e}")
                continue

        # Final save
        pd.DataFrame(results).to_csv(output_file, index=False)
        print("Processing completed!")

def main():
    processor = PromptProcessor(property_questions)
    processor.process_csv('implicit_qa_dataset.csv', 'implicit_qa_results-v2.csv')
    processor.process_csv('explicit_qa_dataset.csv', 'explicit_qa_results-v2.csv')

if __name__ == "__main__":
    main()