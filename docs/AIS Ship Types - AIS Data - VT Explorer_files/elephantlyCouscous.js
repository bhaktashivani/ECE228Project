App = {};
App.main = function(){
    if( 1 === $('nav').children().length )
    {
        $('nav').remove();
        $('.menu-link').remove();
        $('.push').css({
            "width": "100%",
            "margin-left":"0"
        });
        $('.push header').css({
            "width": "100%",
        });
    }

    // Syntax highlighting
    hljs.initHighlightingOnLoad();
}

App.main();
