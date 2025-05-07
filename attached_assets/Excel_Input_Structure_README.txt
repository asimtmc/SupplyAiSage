
# Transition Management Tool – Excel Input Structure

This file outlines the structure required for the input Excel file used by the Transition Management Tool for FMCG operations.

## Expected Excel File Format

The Excel file should contain the following **six sheets**, each with specific column headers and sample formats:

---

### 1. FG Master
| FG Code | Description      | Category   |
|---------|------------------|------------|
| FG1001  | Shampoo 200ml    | Hair Care  |

---

### 2. BOM (Bill of Materials)
| FG Code | Component Code | Component Type | Qty per FG |
|---------|----------------|----------------|------------|
| FG1001  | RM101          | RM             | 0.5        |

---

### 3. FG Forecast
| FG Code | Month   | Forecast Qty |
|---------|---------|---------------|
| FG1001  | 2024-06 | 10000         |

---

### 4. SOH (Stock on Hand)
| Component Code | Component Type | Stock on Hand |
|----------------|----------------|----------------|
| RM101          | RM             | 5000           |

---

### 5. Open Orders
| Component Code | Component Type | Open Order Qty | Expected Arrival |
|----------------|----------------|----------------|------------------|
| RM101          | RM             | 2000           | 2024-06-10       |

---

### 6. Transition Timeline
| FG Code | Old RM/PM    | New RM/PM    | Start Date | Go-Live Date |
|---------|--------------|--------------|------------|---------------|
| FG1001  | RM101/PM201  | RM103/PM203  | 2024-06-01 | 2024-07-01    |

---

## Notes:
- Column names must be **exactly as shown** (case-sensitive).
- Dates should be in `YYYY-MM-DD` format.
- Data validations and error alerts should be implemented in the app for missing/invalid data.

This structure allows the app to simulate PIPO transitions, analyze RM/PM usage, and generate dynamic dashboards for effective transition planning.
