import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    
    // Forward the request to the Flask server
    const flaskResponse = await fetch('http://127.0.0.1:5001/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!flaskResponse.ok) {
      const errorText = await flaskResponse.text();
      return NextResponse.json(
        { error: `Flask server error: ${flaskResponse.status} ${flaskResponse.statusText}`, details: errorText },
        { status: flaskResponse.status }
      );
    }

    const data = await flaskResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Proxy error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to connect to analysis server' },
      { status: 500 }
    );
  }
}

