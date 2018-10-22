// Store inputs from forms
let assetRets, assetVols, corrs, numAssets, asset_weights;

// Store losses at each epoch
let lossArray;

function update(){

    // Fetch data from forms
    eqret = $('#eqret').val();
    bondret = $('#bondret').val();
    goldret = $('#goldret').val();
    cashret = $('#cashret').val();

    eqvol = $('#eqvol').val();
    bondvol = $('#bondvol').val();
    goldvol = $('#goldvol').val();
    cashvol = $('#cashvol').val();

    eqeq = $('#eqeq').val();
    eqbond = $('#eqbond').val();
    eqgold = $('#eqgold').val();
    eqcash = $('#eqcash').val();

    const corr1 = [eqeq, eqbond, eqgold, eqcash];

    bondeq = $('#bondeq').val();
    bondbond = $('#bondbond').val();
    bondgold = $('#bondgold').val();
    bondcash = $('#bondcash').val();

    const corr2 = [bondeq, bondbond, bondgold, bondcash];

    goldeq = $('#goldeq').val();
    goldbond = $('#goldbond').val();
    goldgold = $('#goldgold').val();
    goldcash = $('#goldcash').val();

    const corr3 = [goldeq, goldbond, goldgold, goldcash];

    casheq = $('#casheq').val();
    cashbond = $('#cashbond').val();
    cashgold = $('#cashgold').val();
    cashcash = $('#cashcash').val();

    const corr4 = [casheq, cashbond, cashgold, cashcash];

    flooreq = $('#flooreq').val();
    floorbond = $('#floorbond').val();
    floorgold = $('#floorgold').val();
    floorcash = $('#floorcash').val();

    ceileq = $('#ceileq').val();
    ceilbond = $('#ceilbond').val();
    ceilgold = $('#ceilgold').val();
    ceilcash = $('#ceilcash').val();

    assetRets = [[eqret],
                [bondret],
                [goldret],
                [cashret]
                ];

    assetVols = [[eqvol],
                [bondvol],
                [goldvol],
                [cashvol]
                ];

    corrs = [corr1,
                    corr2,
                    corr3,
                    corr4];

    numAssets = +assetRets.length;

    asset_weights = tf.variable(tf.div(tf.ones([4,1]), numAssets));

    lossArray = [];
}

// Learning rate for the optimiser
const learning_rate = 0.01;

// Number of iterations to run the optimiser; I found that 750 is about enough for the loss (Sharpe Ratio in this case) to stablise
const epochs = 750;

// We use a adam optimiser here. I tried some others like SGD, but adam seems to converge faster
const optimizer = tf.train.adam(learning_rate);

function predict(){
    return tf.tidy(()=>{
    // Normally, we take weightsT*covar*weights
    // Here, we take weightedvolsT*corr*weightedvols
    const weighted_vols = asset_weights.mul(assetVols);
    const vol = tf.sqrt(weighted_vols.transpose().matMul(corrs).matMul(weighted_vols).squeeze());
    const returns_sum = asset_weights.mul(assetRets).sum();
    const sharpe = tf.neg(tf.div(returns_sum,vol));
    // For those not familiar, the Sharpe ratio shows us how much returns we are getting per unit of risk
    // We negate the number as the higher the better, but our optimise is searching for the minimum
    return sharpe;
    });
};

function constraints(){

    // All the constraints that we are applying
    // It looks complicated, but all we are doing is resetting the weights at each cycle
    // Weights that breach the constraints will be set to 0, 1, or the preset floors or ceilings
    const floors = [[flooreq],[floorbond],[floorgold],[floorcash]];

    const mask_wts_less_than_floor = tf.greater( floors, asset_weights )
    const floors_ = asset_weights.assign( tf.where (mask_wts_less_than_floor, tf.tensor(floors), asset_weights) );


    const ceils = [[ceileq],[ceilbond],[ceilgold],[ceilcash]];

    const mask_wts_more_than_ceil = tf.greater( asset_weights, ceils )
    const ceilings_ = asset_weights.assign( tf.where (mask_wts_more_than_ceil, tf.tensor(ceils), asset_weights) );

    const mask_wts_less_than_zero = tf.greater( 0.0, asset_weights )
    const zero_floor = asset_weights.assign( tf.where (mask_wts_less_than_zero, tf.zerosLike(asset_weights), asset_weights) );

    const mask_wts_more_than_one = tf.greater( asset_weights, 1.0 )
    const one_ceil = asset_weights.assign( tf.where (mask_wts_more_than_one, tf.onesLike(asset_weights), asset_weights) )

    // This is a little different
    // Here we make sure the sum of weights = 1 by scaling them down
    const result_sum = tf.sum(asset_weights)
    const reset_equals_one = asset_weights.assign(tf.div(asset_weights, result_sum)) //If sum > 1, dividing it by the sum scales it back to 1

    const returns_sum = tf.mul(asset_weights, assetRets).sum();
    // returns_sum.print();


};

// This helps us to update the progress bar 
let progresspercent;

// The main training function
async function train(epochs, done){
    for (let i=0; i<epochs; i++){
        let cost;

        cost  = tf.tidy(()=>{
            cost = optimizer.minimize(()=>{
                const constraining = constraints();
                const sharpe = predict();
                
                return sharpe;
            },true);
            const constraining = constraints();
            return cost;
        })
        // If we use this line below, we will be pushing in a new point for each iteration
        // cost.data().then((data)=>lossArray.push({i:i, error:-data[0]}));

        progresspercent = 100*i/epochs;
        $('.progress-bar').css('width', progresspercent+'%').attr('aria-valuenow', progresspercent);

        if(i%100==0){
            // await cost.data().then((data)=>console.log(i,data));
            cost.data().then((data)=>lossArray.push({i:i, error:-data[0]}));
            // console.log('Run:', i);
            // cost.print();
            // asset_weights.print();
        }
        await tf.nextFrame();
    }

    // We call the functions in done() later followed by the plotting functions later
    done();

    // Important that we call this here with await to ensure that the operations above are finished first
    // Placing it in the done() function which is not an async function does not help!
    await renderPie(asset_weights_array);
    await ploterrors(lossArray);
    await plot_optimal(finalrets/100,finalvols/100);
}

// Simple computation and update of the Sharpe Ratio
$('#compute').click(()=>{

    $('#eqsharpe').val(Math.round(100*$('#eqret').val()/$('#eqvol').val())+'%');
    $('#bondsharpe').val(Math.round(100*$('#bondret').val()/$('#bondvol').val())+'%');
    $('#goldsharpe').val(Math.round(100*$('#goldret').val()/$('#goldvol').val())+'%');
    $('#cashsharpe').val(Math.round(100*$('#cashret').val()/$('#cashvol').val())+'%');

});

// Save all the results here

let final_asset_weights, final_returns;

let asset_weights_array=[];

let asset_names = ['Equities', 'Bonds', 'Gold', 'Cash'];

for(let i=0; i<asset_names.length; i++){
    asset_weights_array.push({Asset:asset_names[i], Proportion:0.25});
}

// console.log('Before:', asset_weights_array);

let finalrets, finalvols;

// Train once the optimise button is clicked

$('#optimise').click(()=>{

    update();
    train(epochs, ()=>{
        // console.log('Training Completed');

        asset_weights_array = [];

        let final_asset_weights= asset_weights.dataSync();
        // console.log(final_asset_weights);

        for(let i=0; i<asset_names.length; i++){
            asset_weights_array.push({Asset:asset_names[i], Proportion:final_asset_weights[i]});
        }
        // console.log('After:', asset_weights_array);


        // Write the final return and vol of the portfolio to the screen 
        weighted_vols = asset_weights.mul(assetVols);
        vol = weighted_vols.transpose().matMul(corrs).matMul(weighted_vols).squeeze();
        finalvols = 100*Math.sqrt(vol.dataSync());
        $('#pfvol').text(finalvols.toFixed(2)+'%');

        returns_sum = asset_weights.mul(assetRets).sum();
        finalrets = 100*returns_sum.dataSync();
        $('#pfreturn').text(finalrets.toFixed(2)+'%');
        sharpe = tf.neg(tf.div(returns_sum,vol));

        

        // This will get called before the operations above are completed
        // render(asset_weights_array);
    });
    
});

// D3 Chart

// Setup the pie chart
// I should probably enclose this in a function like the subsequent charts but that's for another day
const formatNum = d3.format(".2f");

const width = 350;
const height = 350;

const radius = Math.min(width,height)/4;

const color = d3.scaleOrdinal(d3.schemeCategory20);

let svg = d3.select('#d3canvas')
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', 'translate('+(width/2)+','+(height/2)+')');



let arc = d3.arc()
            .innerRadius(0)
            .outerRadius(radius);

let outerArc = d3.arc()
                .innerRadius(radius)
                .outerRadius(radius+100);

let pie = d3.pie()
            .value(function(d){return d.Proportion;})
            .sort(null);     

let path = svg.selectAll("path")
            .data(pie(asset_weights_array))
            .enter().append("path")
            .attr("class","piechart")
            .attr("fill", function(d,i){ return color(i); })
            .attr("fill-opacity", 0.5)
            .attr("d", arc)
            .each(function(d){ this._current = d; });

let text = svg.selectAll("text")
            .data(pie(asset_weights_array))
            .enter().append('text')
            .attr("d", arc)
            .attr('transform', function(d){
                return 'translate(' + outerArc.centroid(d) +')';
            })
            .text(function(d){return d.data.Asset +' | ' + 100*d.data.Proportion+'%'}) 
            .each(function(d) { this._current = d; })
            .style('text-anchor', 'middle')
            .style('font-size', '8px');




function renderPie(data){
    // console.log('Rendering');

        svg.selectAll("path").data(pie(data)).transition().duration(500).attrTween("d", arcTween)
        // add any new paths
        svg.selectAll("path")
        .data(pie(data))
        .enter().append("path")
        .attr("class","piechart")
        .attr("fill", function(d,i){ return color(i); })
        .attr("d", arc)
        .each(function(d){ this._current = d; })

        // remove data not being used
        svg.selectAll("path")
        .data(pie(data)).exit().remove();

        // Do the same for text
        // Here we tween and move existing text
        svg.selectAll("text")
        .data(pie(data))
        .transition()
        .duration(500)
        .attrTween("transform", textTween)
        .text(function(d){return d.data.Asset +' | ' + Math.floor(100*d.data.Proportion)+'%'})

        // Add new text
        svg.selectAll("text")
        .data(pie(data))
        .enter().append('text')
        .attr("d", arc)
        .attr('transform', function(d){
            return 'translate(' + outerArc.centroid(d) +')';
        })
        .text(function(d){return d.data.Asset +' | ' + Math.floor(100*d.data.Proportion)+'%'})
        .each(function(d) { this._current = d; })
        .style('text-anchor', 'middle')
        .style('font-size', '8px');

        // remove data not being used
        svg.selectAll("text")
        .data(pie(data)).exit().remove();
}


// Tween the arc paths
// Interpolate between the current to the next point
function arcTween(a) {
  let i = d3.interpolate(this._current, a);
  this._current = i(0);
  return function(t) {
    return arc(i(t));
  };
}

function midAngle(d){
    return d.startAngle + (d.endAngle - d.startAngle)/2;
}

// Interpolate between the current position to the next point
function textTween(a){
    this._current = this._current || a;
    let interpolate = d3.interpolate(this._current, a);
    this._current = interpolate(0);
    return function(t) {
        var d2 = interpolate(t);
        var pos = outerArc.centroid(d2);
        // pos[0] = radius * (midAngle(d2) < Math.PI ? 1 : -1);
        return "translate("+ pos +")";
    }    
}

// Set up constants for the chart to plot the losses (Sharpe Ratio)
// I call it errors here as this is the equivalent of errors in other models
let margin = 50;
let svg2;
let errorline;
let scaleX2, scaleY2;
let xAxis2, yAxis2;

let xAxisGroup2, yAxisGroup2;

// Setup the chart for plotting errors

function setup_ploterrors(){

    // console.log('Set up errors');
    svg2 = d3.select('#d3canvas2')
                .append('svg')
                .attr('width',width)
                .attr('height',height);


    scaleX2 = d3.scaleLinear().range([2*margin,width-2*margin]);
    scaleY2 = d3.scaleLinear().range([height-2*margin,2*margin]);

    xAxis2 = d3.axisBottom();
    yAxis2 = d3.axisLeft();

    xAxis2.scale(scaleX2).ticks(5);
    yAxis2.scale(scaleY2).ticks(5).tickFormat(d3.format(1));

    xAxisGroup2 = svg2.append('g')
                          .attr('class', 'x axis')
                          .attr('transform', 'translate(0' + ',' + (height-2*margin) + ')');

    yAxisGroup2 = svg2.append('g')
                          .attr('class', 'y axis')
                          .attr('transform', 'translate('+ 2*margin + ',0)');

    xAxisGroup2.call(xAxis2);
    yAxisGroup2.call(yAxis2); 

    // Plot title
    svg2.append('text')
        .attr('x', width/2)
        .attr('y', 40)
        .style('text-anchor', 'middle')
        .style('fill', '#FFFFFF')
        .style('font-size', '10px')
        .text('Training Chart');
    
    // X and Y axis labels
    svg2.append('text')
        .attr('x', width/2)
        .attr('y', height-65)
        .style('text-anchor', 'middle')
        .style('fill', '#FFFFFF')
        .style('font-size', '10px')
        .text('Epochs');

    svg2.append('text')
        .attr('x', 45)
        .attr('y', height/2)
        .style('text-anchor', 'middle')
        .style('fill', '#FFFFFF')
        .style('font-size', '10px')
        .text('Sharpe');      

    // We draw a line here too, instead of just circles/dots above
    errorline = d3.line()
                        .curve(d3.curveCatmullRom)
                        .x(function(d){return scaleX2(d.i);})
                        .y(function(d){return scaleY2(d.error);});


}

// Function to plot errors 

function ploterrors(errors){

    // console.log('Plot errors');

    // console.log(errors);

    scaleX2.domain(d3.extent(errors, function(d){return d.i}));
    scaleY2.domain(d3.extent(errors, function(d){return d.error}));

    let errordots = svg2.selectAll('errordots')
                    .data(errors);

        errordots
                    .enter()
                    .append('circle')
                    .style('fill', 'none')
                    .attr('cx',0)
                    .attr('cy',height/2)
                    .merge(errordots)                
                    .transition()
                    .duration(200)
                    .attr('class', 'circle')
                    .attr('cx',d=>scaleX2(d.i))
                    .attr('cy',d=>scaleY2(d.error))
                    .attr('r', 1)
                    .style('fill', '#F92A82')
                    .style('fill-opacity', 0.5);
                    

        errordots.exit().remove();


    let errorsline = svg2.append('path')
                        .data([errors])
                        .attr('class', 'line')
                        .attr('d', errorline)
                        .style('fill', 'none')
                        .style('stroke', 'pink')
                        .style('stroke-width', '2px');

        xAxis2.scale(scaleX2).ticks(5);
        yAxis2.scale(scaleY2).ticks(5).tickFormat(d3.format(1));

        xAxisGroup2.call(xAxis2);
        yAxisGroup2.call(yAxis2); 
}

// The subsequent lines here are to plot the dot charts of all possible portfolio combinations
// So that we can see the efficient frontier
$('#generate').click(()=>{
    update();
    let efdata = generateEfficientFrontier(1501);
    plot_ef(efdata);
});

function generateEfficientFrontier(numIterations){
    // let num_portfolios = 4;

    asset_array = [];

    for (let i=0; i<numIterations; i++){
        let eqwt = Math.random();
        let bondwt = Math.random();
        let goldwt = Math.random();
        let cashwt = Math.random();
        let sumwt = eqwt + bondwt + goldwt + cashwt;
        // console.log(eqwt/sumwt, bondwt/sumwt, goldwt/sumwt, cashwt/sumwt);
        // console.log(eqwt/sumwt+bondwt/sumwt+goldwt/sumwt+cashwt/sumwt)

        let random_assetwt = tf.tensor([[eqwt/sumwt],
                      [bondwt/sumwt],
                      [goldwt/sumwt],
                      [cashwt/sumwt]]);

        // random_assetwt.print();
        let weighted_vols = random_assetwt.mul(assetVols);
        let vol = tf.sqrt(weighted_vols.transpose().matMul(corrs).matMul(weighted_vols).squeeze());
        let returns_sum = random_assetwt.mul(assetRets).sum();

        // vol.print();
        // returns_sum.print(); 

        asset_array.push({weights:random_assetwt.dataSync(), ret:returns_sum.dataSync()[0], vol:vol.dataSync()[0]});

        if(i%100==0){
            progresspercent = 100*i/numIterations;
            // console.log(progresspercent);
            $('.progress-bar').css('width', progresspercent+'%').attr('aria-valuenow', progresspercent);            
        }

    }

// console.log(asset_array[0].weights);
return asset_array;
}

let svg3;
let scaleX3, scaleY3;
let xAxis3, yAxis3;

let xAxisGroup3, yAxisGroup3;

function setup_ef(){

    // console.log('Set up EF');
    svg3 = d3.select('#d3canvas3')
                .append('svg')
                .attr('width',width)
                .attr('height',height);

    // let scaleX2 = d3.scaleLinear().range([2*margin,width-2*margin]).domain(d3.extent(errors,(d,i)=>i));
    // let scaleY2 = d3.scaleLinear().range([height-2*margin,2*margin]).domain(d3.extent(errors,(d,i)=>+d));

    scaleX3 = d3.scaleLinear().range([2*margin,width-2*margin]);
    scaleY3 = d3.scaleLinear().range([height-2*margin,2*margin]);

    xAxis3 = d3.axisBottom();
    yAxis3 = d3.axisLeft();

    xAxis3.scale(scaleX3).ticks(5);
    yAxis3.scale(scaleY3).ticks(5);

    xAxisGroup3 = svg3.append('g')
                          .attr('class', 'x axis')
                          .attr('transform', 'translate(0' + ',' + (height-2*margin) + ')');

    yAxisGroup3 = svg3.append('g')
                          .attr('class', 'y axis')
                          .attr('transform', 'translate('+ 2*margin + ',0)');

    xAxisGroup3.call(xAxis3);
    yAxisGroup3.call(yAxis3); 

    // Plot title
    svg3.append('text')
        .attr('x', width/2)
        .attr('y', 40)
        .style('text-anchor', 'middle')
        .style('fill', '#FFFFFF')
        .style('font-size', '10px')
        .text('Efficient Frontier');
    
    // X and Y axis labels
    svg3.append('text')
        .attr('x', width/2)
        .attr('y', height-65)
        .style('text-anchor', 'middle')
        .style('fill', '#FFFFFF')
        .style('font-size', '10px')
        .text('Vol');

    svg3.append('text')
        .attr('x', 45)
        .attr('y', height/2)
        .style('text-anchor', 'middle')
        .style('fill', '#FFFFFF')
        .style('font-size', '10px')
        .text('Returns');      

}



function plot_ef(data){

    // console.log('Plot EF');

    // console.log(data);

    scaleX3.domain(d3.extent(data, function(d){return d.vol}));
    scaleY3.domain(d3.extent(data, function(d){return d.ret}));

    let efdots = svg3.selectAll('errordots')
                    .data(data);

        efdots
                    .enter()
                    .append('circle')
                    .style('fill', 'none')
                    .attr('cx',0)
                    .attr('cy',height/2)
                    .merge(efdots)                
                    .transition()
                    .duration(200)
                    .attr('class', 'circle')
                    .attr('cx',d=>scaleX3(d.vol))
                    .attr('cy',d=>scaleY3(d.ret))
                    .attr('r', 1)
                    .style('fill', '#F92A82')
                    .style('fill-opacity', 0.5);
                    

        efdots.exit().remove();


        xAxis3.scale(scaleX3).ticks(5);
        yAxis3.scale(scaleY3).ticks(5);

        xAxisGroup3.call(xAxis3);
        yAxisGroup3.call(yAxis3); 
}

function plot_optimal(ret,vol){

    // console.log('Plot Optimal');

    // console.log(ret,vol);

    let optdots = svg3
                    .append('circle')
                    .attr('class', 'optcircle')
                    .attr('cx',scaleX3(vol))
                    .attr('cy',scaleY3(ret))
                    .attr('r', 3)
                    .style('fill', 'steelblue')
                    .style('fill-opacity', 0.7);
                    
}

// Initialise all the charts
setup_ploterrors();
renderPie(asset_weights_array);
setup_ef();