# Artificial inteligence with pytorch examples
This is my collection of examples. The examples are located in independent folders under ```/src```, and they can be excecuted like this ```python /src/*/main.py```

## predict salary when you find out of a new family member
* src/salary.py

## optimize pimpmydrawing SALES with AI

### data
Drawings extracted from Laravel with the following query

```
collect(
    App\File::all()->unique('name')->map(
        function($file) {
            $file = $file->only(['name', 'updated_at', 'downloads']);
            $file["updated_at"] = $file["updated_at"]->timestamp;
            return $file;
        })->unique('name')->values()->all()
)->shuffle()->toJSON()
```
Also delete id=1 (it got 500K downloads spike.)

#### questions
* Why does SGD work? The data is not convex, but the algorithm adds stocastic global minimum search?
* How to properly normalize/unnormalize data?
* How to setup databases as input source?
* How swap this example to working on the GPU?
* General improvements


#

* Search terms extracted from google search console

