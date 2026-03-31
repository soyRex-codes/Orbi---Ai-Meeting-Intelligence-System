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

---

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