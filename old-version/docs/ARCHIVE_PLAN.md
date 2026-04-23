# Files to Archive in old-version/ Folder

## Background
The project has been simplified from a 3-level multi-model architecture to a single baseline pipeline. The following files/folders are no longer needed and should be archived.

---

## Files & Folders to Move to old-version/

### Directory Structure
```
old-version/
├── configs/                    # Old multi-level config files (no longer used)
├── reports/                    # Old report templates
├── README_Sanjeev.md          # Original notes (kept for reference)
└── (any old model implementations if present)
```

### Detailed File List

#### 1. configs/ (Entire directory)
**Location:** `project-root/configs/`  
**Reason:** Simplified project uses inline CLI parameters instead of YAML configs  
**Action:** Move entire `configs/` folder to `old-version/configs/`

**Contents to move:**
- `level1.yaml` (old Level 1 config)
- `level2.yaml` (old Level 2 config - no longer used)
- `level3.yaml` (old Level 3 config - no longer used)

---

#### 2. reports/ (Entire directory)
**Location:** `project-root/reports/`  
**Reason:** Old result templates; new results go directly to `outputs/`  
**Action:** Move entire `reports/` folder to `old-version/reports/`

**Contents to move:**
- `results_summary.md` (old template)

---

#### 3. README_Sanjeev.md
**Location:** `project-root/README_Sanjeev.md`  
**Reason:** Original project documentation; superseded by new simplified README.md  
**Action:** Move to `old-version/README_Sanjeev.md`

---

#### 4. Old Model Files (if present in src/models/)
**Location:** `project-root/src/models/`  
**Reason:** No longer used (single baseline only)  
**Files to move if present:**
- `level2_finetune_ldm.py` (not used anymore)
- `level3_advanced.py` (not used anymore)
- `level1_sd_inpaint.py` (if replaced)

**Action:** Move to `old-version/models/`

---

## Files to KEEP (Do Not Move)

### Core Implementation
- ✅ `src/data/dataset.py` — Keep (simplified but functional)
- ✅ `src/eval/metrics.py` — Keep (PSNR/SSIM computation)
- ✅ `src/eval/visualize.py` — Keep (4-panel grids)

### Core Scripts
- ✅ `scripts/01_download_images.py` — Keep
- ✅ `scripts/02_prepare_data.py` — Keep
- ✅ `scripts/03_run_inpainting.py` — Keep (main pipeline)

### Documentation
- ✅ `README.md` — Keep (simplified version)
- ✅ `MID_SEMESTER.md` — Keep (all 5 requirements)
- ✅ `project-details.md` — Keep (task reference)
- ✅ `SIMPLIFIED_PROJECT.md` — Keep (project plan)

### Configuration
- ✅ `requirements.txt` — Keep (updated with minimal deps)

### Data & Output
- ✅ `data/` — Keep (will be auto-populated)
- ✅ `outputs/` — Keep (results directory)

---

## Manual Archive Steps

```bash
# 1. Create old-version directory
mkdir old-version

# 2. Move config files
mv configs/ old-version/

# 3. Move reports
mv reports/ old-version/

# 4. Move old documentation
mv README_Sanjeev.md old-version/

# 5. If old model files exist, move them
mv src/models/level2_finetune_ldm.py old-version/models/  # if exists
mv src/models/level3_advanced.py old-version/models/      # if exists

# 6. Verify structure
ls -la old-version/
# Should show: configs/, reports/, README_Sanjeev.md, possibly models/
```

---

## Verification Checklist

After archiving, verify:
- [ ] `configs/` removed from project root
- [ ] `reports/` removed from project root
- [ ] `README_Sanjeev.md` removed from project root
- [ ] All other files remain in place
- [ ] `old-version/` folder contains archived files
- [ ] `scripts/01_*.py`, `02_*.py`, `03_*.py` present and unmodified
- [ ] `src/eval/` and `src/data/` present and simplified

---

## Notes
- The project is now self-contained in 3 scripts
- All outputs go to `outputs/` directory (auto-created)
- No configuration files needed (CLI parameters instead)
- MID_SEMESTER.md contains all 5 required sections
- README.md is focused on quick start and execution
