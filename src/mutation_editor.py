import run


def add_fig_clickable_mutation_editor(fig, record, outdir):

    def add_click_callback(fig):
        """
        Add a callback to the figure that logs clicked data points to the console.

        :param fig: A plotly.graph_objects.Figure object
        :return: A Figure with the callback added
        """
        # Add clickmode to the layout
        fig.update_layout(clickmode="event+select")
        return fig

    # Add click callback
    interactive_fig = add_click_callback(fig)

    # Save the interactive figure to an HTML file
    html_file = f"{outdir}/mutation_editor.html"
    run.plotly_fig_to_html(interactive_fig, html_file, include_plotlyjs=True)

    # Read the HTML file
    with open(html_file, "r") as file:
        html_content = file.read()

    js_to_inject = """
    <style>
        body {
            font-family: 'Open Sans', Arial, sans-serif;
            background-color: #ffffff;
            color: #444444;
            line-height: 1.6;
        }
        #mutation-generator {
            max-width: 1260px;
            margin: 20px 0 0 40px;
            padding: 20px;
            background-color: #f8f8f8;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #2f3136;
            font-size: 24px;
            margin-bottom: 20px;
            text-align: left;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #666666;
            font-weight: 600;
        }
        textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            font-size: 14px;
            resize: vertical;
            transition: border-color 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #7f7f7f;
        }
        .button-container {
            text-align: center;
            margin-top: 20px;
        }
        .button {
            color: white;
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 400;
            transition: background-color 0.3s;
            display: inline-block;
            margin: 0 10px;
        }
        #validateButton {
            background-color: #5cb85c;
        }
        #validateButton:hover {
            background-color: #4cae4c;
        }
        #resetButton {
            background-color: #337ab7;
        }
        #resetButton:hover {
            background-color: #286090;
        }
    </style>

    <div id="mutation-generator">
        <h1>Type mutations above to get FASTA</h1>
        <label for="mutations">Mutations (one set per line):</label>
        <textarea id="mutations" placeholder="M1A" rows="6" oninput="generateMutations()"></textarea>
        <br>
        <label for="output">FASTA:</label>
        <textarea id="output" readonly rows="6"></textarea>
        <br>
        <div class="button-container">
            <button id="validateButton" class="button" onclick="validateWithImmunoGeNN()">Predict with ImmunoGeNN</button>
            <button id="resetButton" class="button" onclick="resetMutations()">Reset mutations</button>
        </div>
    </div>

    <script>
    const fixedHeader = ">B8V31";
    const fixedSequence = "EVQLVESGGGVVQPGGSLRLSCAASGEIKSINFMRWYRQAPGKQREWVAGFTRDGSTNYPDSAKGRFTISRDNAKNTVYLQIDSLKPEDTAVYYCYMLDTWGQGTQVTVSS";

    function generateMutations() {
        const mutationsInput = document.getElementById('mutations').value;
        const outputTextarea = document.getElementById('output');
        const mutationSets = mutationsInput.split('\\n').filter(m => m.trim() !== '');
        let output = `>${fixedHeader}\\n${fixedSequence}\\n`;
        for (const mutationSet of mutationSets) {
            const mutations = mutationSet.split(/\\s+/);
            if (mutations.every(isValidMutation)) {
                let mutatedSequence = fixedSequence;
                let allValid = true;
                for (const mutation of mutations) {
                    const position = parseInt(mutation.slice(1, -1)) - 1;
                    const wildTypeResidue = mutation[0];
                    const newAminoAcid = mutation.slice(-1);
                    if (mutatedSequence[position] === wildTypeResidue) {
                        mutatedSequence =
                            mutatedSequence.slice(0, position) +
                            newAminoAcid +
                            mutatedSequence.slice(position + 1);
                    } else {
                        allValid = false;
                        break;
                    }
                }
                if (allValid) {
                    output += `>${fixedHeader}__${mutations.join('_')}\\n${mutatedSequence}\\n`;
                }
            }
        }
        outputTextarea.value = output.trim();
    }

    function isValidMutation(mutation) {
        const regex = /^[A-Z]\\d+[A-Z]$/;
        if (!regex.test(mutation)) return false;
        const position = parseInt(mutation.slice(1, -1)) - 1;
        return position >= 0 && position < fixedSequence.length;
    }

    function appendMutation(mutation) {
        const mutationsTextarea = document.getElementById('mutations');
        const currentContent = mutationsTextarea.value;
        mutationsTextarea.value = currentContent + (currentContent ? '\\n' : '') + mutation;
        generateMutations();
    }

    function validateWithImmunoGeNN() {
        const fastaContent = document.getElementById('output').value;
        const escapedFastaContent = fastaContent.replace(/\\n/g, '\\\\n');
        const jsonString = `{"--fasta_file":"${escapedFastaContent}"}`;
        const base64String = btoa(jsonString);
        const url = `https://biolib.com/DTU/ImmunoGeNN/#input=${base64String}`;
        window.open(url, '_blank');
    }

    function resetMutations() {
        document.getElementById('mutations').value = '';
        generateMutations();
    }

    // Add click event listener to the plot
    var plot = document.getElementsByClassName('plotly-graph-div')[0];
    plot.on('plotly_click', function(data) {
        var clickedX = data.points[0].x;
        var allPoints = data.points;
        
        var pointsWithSameX = allPoints.filter(function(point) {
            return point.x === clickedX;
        });
        
        pointsWithSameX.forEach(function(point) {
            if (point.customdata && point.customdata.length > 5) {
                var pointData = point.customdata[4];
                var isTrue = point.customdata[5];
                if (isTrue === true) {
                    appendMutation(pointData);
                }
            }
        });
    });

    // Initialize the output with the original sequence
    document.getElementById('output').value = `${fixedHeader}\\n${fixedSequence}`;
    generateMutations();
    </script>
    """

    # Replace sequence
    orig_str = 'const fixedSequence = "EVQLVESGGGVVQPGGSLRLSCAASGEIKSINFMRWYRQAPGKQREWVAGFTRDGSTNYPDSAKGRFTISRDNAKNTVYLQIDSLKPEDTAVYYCYMLDTWGQGTQVTVSS";'
    replace_str = f'const fixedSequence = "{record.sequence}";'
    js_to_inject = js_to_inject.replace(orig_str, replace_str)

    # Replace ID
    orig_str = 'const fixedHeader = ">B8V31";'
    replace_str = f'const fixedHeader = "{record.id}";'
    js_to_inject = js_to_inject.replace(orig_str, replace_str)

    # Inject the JavaScript just before the closing </body> tag
    html_content = html_content.replace("</body>", f"{js_to_inject}</body>")

    # Write the modified HTML back to the file
    with open(html_file, "w") as file:
        file.write(html_content)

    # print(f"HTML file has been saved as '{html_file}' with click logging functionality.")
