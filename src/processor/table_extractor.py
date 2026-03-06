import pandas as pd
from typing import Dict, Any, List

class TableExtractor:
    """Extract and process tables from documents."""
    
    def extract(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert table to structured format with summary."""
        # In production, use unstructured.io or camelot for better extraction
        
        if not table_data or not table_data.get("data"):
            return {"dataframe": None, "summary": ""}
        
        # Create DataFrame with a simple header heuristic
        df = self._to_dataframe(table_data.get("data", []))
        
        # Generate text summary
        summary = self._generate_summary(df)
        
        return {
            "dataframe": df.to_dict(),
            "summary": summary,
            "shape": [int(df.shape[0]), int(df.shape[1])]
        }

    def _to_dataframe(self, rows: List[List[Any]]) -> pd.DataFrame:
        """Build DataFrame from extracted rows while preserving likely headers."""
        if not rows:
            return pd.DataFrame()

        first_row = rows[0]
        likely_header = (
            len(rows) > 1
            and all(isinstance(cell, str) and cell.strip() for cell in first_row)
            and len(set(first_row)) == len(first_row)
        )

        if likely_header:
            return pd.DataFrame(rows[1:], columns=first_row)
        return pd.DataFrame(rows)
    
    def _generate_summary(self, df: pd.DataFrame) -> str:
        """Generate natural language summary of table."""
        if df.empty:
            return "Empty table"
        
        rows, cols = df.shape
        summary = f"Table with {rows} rows and {cols} columns. "
        
        # Add column names
        if not df.columns.empty:
            summary += f"Columns: {', '.join(map(str, df.columns))}. "
        
        # Add sample data
        if rows > 0:
            summary += f"First row: {df.iloc[0].to_dict()}"
        
        return summary
