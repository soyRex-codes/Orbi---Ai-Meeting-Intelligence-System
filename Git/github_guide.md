# Git new branch and best practices steps

```bash
# Step 1 — Create Your Branch
# Make sure you start from latest main
git checkout main #check if on branch main, if not switch to it
git pull origin main #pull latest main to your local

# Create your new branch and switch to it
git checkout -b feature/branch_name  # create and switch to new branch, name it something descriptive - do only once when creating the branch
git push -u origin feature/branch_name #push your new branch to GitHub and set upstream only do once when creating the branch
```

# Step 2 — Work on Your Branch Daily
# Start of every work session
git pull origin feature/branch_name   # Pull your own branch updates
#possible merge conflicts to deal with here if you haven't pulled in a while

# End of work session — save your progress
git add .
git commit -m "feat: added whisper medium model transcription"
git push origin feature/branch_name

# Step 3 — Keep Your Branch Up to Date with Main
git checkout main                            # Switch to main
git pull origin main                         # Get latest main
git checkout feature/branch_name     # Switch back to your branch
git merge main                               # Bring main's changes into your branch

# ##############################
# How to enter the message in Vim:
Enter "Insert" mode: Press the i key on your keyboard. You should see -- INSERT -- appear at the bottom of the screen.
Type your message: Use the arrow keys to move to the very first line (above the lines starting with #) and type your merge reason.
Exit "Insert" mode: Press the Esc key once. The -- INSERT -- text at the bottom will disappear.
Save and Close: Type :wq (the colon is required) and press Enter.
# ##############################
---






# Before You Start Working — Always Pull First
git stash                    # Save your local changes temporarily
git pull origin main         # Pull latest from teammate
git stash pop                # Reapply your changes on top

# When You're Ready to Push
git stash                    # Stash local changes
git pull origin main         # Pull any new teammate changes
git stash pop                # Reapply your changes
git add .                    # Stage everything
git commit -m "your message" # Commit
git push origin main         # Push to GitHub
```

### The Golden Rules

> 1. Always stash → pull → stash pop before pushing
> 2. Pull at the start of every work session — not just when pushing
> 3. Commit often — small commits are easier to merge than big ones

---

### Save This as a Cheatsheet
```
START OF DAY          PUSH YOUR WORK
──────────────        ──────────────
git stash             git stash
git pull origin main  git pull origin main
git stash pop         git stash pop
  ↓                   git add .
start coding          git commit -m "message"
                      git push origin main