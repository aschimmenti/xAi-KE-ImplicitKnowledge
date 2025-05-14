import pandas as pd
from openai import OpenAI
import time
import json
from typing import List, Dict

class PromptProcessor:
    def __init__(self, batch_size: int = 10, max_retries: int = 3, retry_delay: int = 60):
        self.client = OpenAI()
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def get_last_processed_id(self, output_file: str) -> str:
        try:
            df_output = pd.read_csv(output_file, encoding='utf-8')
            if not df_output.empty:
                last_id = df_output.iloc[-1]['id']
                print(f"\nFound last processed ID: {last_id}")
                return last_id
        except FileNotFoundError:
            print("\nNo existing output file found, starting from beginning")
        return None

    def process_prompt(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                print(f"\nAttempt {attempt + 1}/{self.max_retries}")
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "text"},
                    temperature=0,
                    max_completion_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                generated_text = response.choices[0].message.content
                print("\nGenerated Response:")
                print("-" * 50)
                print(generated_text)
                print("-" * 50)
                return generated_text
            except Exception as e:
                if "connection" in str(e).lower() and attempt < self.max_retries - 1:
                    print(f"\nConnection error: {e}")
                    print(f"Attempt {attempt + 1}/{self.max_retries}. Waiting {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                print(f"\nError processing prompt: {e}")
                return f"Error: {str(e)}"
        print("\nMax retries exceeded")
        return "Error: Max retries exceeded"

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in batch.iterrows():
            try:
                print(f"\nProcessing prompt for ID: {row['id']}")
                print(f"Selected property: {row['selected_property']}")
                print(f"Property value: {row['property_value']}")
                generated_text = self.process_prompt(row['prompt'])
                row_dict = row.to_dict()
                row_dict['generated_text'] = generated_text
                results.append(row_dict)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error processing row: {e}")
                row_dict = row.to_dict()
                row_dict['generated_text'] = f"Error: {str(e)}"
                results.append(row_dict)
        return pd.DataFrame(results)

    def process_csv(self, input_file: str, output_file: str) -> None:
        try:
            last_processed_id = self.get_last_processed_id(output_file)
            df = pd.read_csv(input_file, encoding='utf-8')
            
            if last_processed_id:
                start_idx = df[df['id'] == last_processed_id].index
                if len(start_idx) > 0:
                    df = df.iloc[start_idx[0] + 1:]
                    print(f"Resuming from ID: {last_processed_id}")

            total_rows = len(df)
            all_results = []

            # Load existing results if any
            try:
                existing_results = pd.read_csv(output_file, encoding='utf-8')
                all_results = [existing_results]
            except FileNotFoundError:
                pass

            for i in range(0, total_rows, self.batch_size):
                batch = df.iloc[i:i + self.batch_size]
                print(f"Processing batch {i//self.batch_size + 1} of {(total_rows + self.batch_size - 1)//self.batch_size}")
                batch_results = self.process_batch(batch)
                all_results.append(batch_results)
                combined_results = pd.concat(all_results, ignore_index=True)
                combined_results.to_csv(output_file, index=False)
                print(f"Saved progress: {len(combined_results)} rows processed")

            print("Processing completed!")
            
        except Exception as e:
            print(f"Error processing CSV: {e}")
            raise

def main():
    processor = PromptProcessor(batch_size=10)
    try:
        processor.process_csv('explicit_entity_prompts.csv', 'explicit_output_prompts.csv')
    except Exception as e:
        print(f"Processing failed: {e}")

if __name__ == "__main__":
    main()