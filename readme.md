# predict salary when you find out of a new family member
* src/salary.py

# optimize pimpmydrawing SALES with AI

## data
* Drawings extracted from Laravel

```
collect(App\File::all()->unique('name')->map(function($file) { $file = $file->only(['name', 'updated_at', 'downloads']); $file["updated_at"] = $file["updated_at"]->timestamp; return $file; })->unique('name')->values()->all())->toJSON()
```

* Search terms extracted from google search console

